"""
DILI-PLUS | DILI 代理标签与观察终点生成（包实现）

职责：将“基线状态正常/偏低，后续 ALT 或 AST 数值达到 120 U/L”定义为
DILI 代理阳性；阳性记录在首次达阈时间截断，阴性记录在用药时间边界内随机截断。
输入：data_cache/01_aligned_dili_labs.parquet。
输出：data_cache/02_dili_labels_censored.parquet。
状态：当前 DILI 单任务数据流程的第二步。
安全：源数据库连接统一通过只读工厂，固定使用 read_only=True。
"""

import os
import pandas as pd
import time

from diliplus.config import load_settings
from diliplus.database import connect_source_database

def extract_dili_labels_and_censor(settings=None):
    settings = settings or load_settings()
    data_dir = str(settings.paths.data_cache)
    
    input_parquet = os.path.join(data_dir, "01_aligned_dili_labs.parquet")
    output_parquet = os.path.join(data_dir, "02_dili_labels_censored.parquet")
    
    print(f"🔗 [DILIPLUS] Connecting to database and loading tensor: {input_parquet}")
    conn = connect_source_database(settings)
    start_time = time.time()

    print("\n⏳ [DILIPLUS] Executing State Transitions & Probabilistic Right-Censoring...")
    
    label_query = f"""
    CREATE OR REPLACE TEMP TABLE temp_dili_labels AS
    WITH RankedLabs AS (
        SELECT 
            encounter_id,
            first_med_time,
            last_med_time,
            abnormal_status,
            lab_value,
            lab_time,
            ROW_NUMBER() OVER(PARTITION BY encounter_id ORDER BY lab_time ASC) as seq_num
        FROM read_parquet('{input_parquet}')
        WHERE lab_item LIKE '%谷丙转氨酶%' OR lab_item LIKE '%谷草转氨酶%'
    ),
    PatientSummary AS (
        SELECT 
            encounter_id,
            MAX(first_med_time) AS first_med_time,
            MAX(last_med_time) AS last_med_time,
            MAX(CASE WHEN seq_num = 1 THEN abnormal_status END) as baseline_status,
            -- DILI 阳性代理标准：后续化验飙高 (>120 为强异常信号的重症替代标准)
            MAX(CASE WHEN seq_num > 1 AND TRY_CAST(lab_value AS FLOAT) >= 120.0 THEN 1 ELSE 0 END) as has_subsequent_high,
            MIN(CASE WHEN seq_num > 1 AND TRY_CAST(lab_value AS FLOAT) >= 120.0 THEN lab_time END) as t_onset,
            COUNT(*) as lab_frequency
        FROM RankedLabs
        GROUP BY encounter_id
    ),
    RawLabels AS (
        SELECT 
            encounter_id,
            first_med_time,
            last_med_time,
            lab_frequency,
            CASE 
                WHEN baseline_status IN ('N', '正常', 'L', '低', '↓') AND has_subsequent_high = 1 THEN 1
                WHEN baseline_status IN ('N', '正常', 'L', '低', '↓') AND has_subsequent_high = 0 THEN 0
                ELSE NULL 
            END AS label_dili,
            t_onset
        FROM PatientSummary
    )
    SELECT 
        encounter_id,
        label_dili,
        t_onset,
        first_med_time,
        last_med_time,
        lab_frequency,
        -- 🌟 DILIPLUS 核心架构：有界概率性右删失 (Bounded Probabilistic Right-Censoring)
        CASE 
            WHEN label_dili = 1 THEN t_onset  -- 阳性组：严格在发病时刻截断
            WHEN label_dili = 0 THEN 
                -- 阴性组：在物理住院边界内，生成符合均匀分布的随机物理截断点
                TO_TIMESTAMP(
                    EPOCH(first_med_time) + 
                    RANDOM() * (EPOCH(last_med_time) - EPOCH(first_med_time))
                )
            ELSE NULL 
        END AS censor_time
    FROM RawLabels
    WHERE label_dili IS NOT NULL;
    """
    
    try:
        conn.execute(label_query)
        conn.execute(f"COPY temp_dili_labels TO '{output_parquet}' (FORMAT PARQUET)")
        
        df_stats = conn.execute("""
            SELECT 
                COUNT(*) as valid_patients,
                SUM(CASE WHEN label_dili = 1 THEN 1 ELSE 0 END) as dili_positive,
                SUM(CASE WHEN label_dili = 0 THEN 1 ELSE 0 END) as dili_negative,
                AVG(CASE WHEN label_dili = 1 THEN EPOCH(censor_time) - EPOCH(first_med_time) END)/3600 AS avg_pos_window_hours,
                AVG(CASE WHEN label_dili = 0 THEN EPOCH(censor_time) - EPOCH(first_med_time) END)/3600 AS avg_neg_window_hours
            FROM temp_dili_labels;
        """).df()
        
        print(f"✅ [DILIPLUS] Bounded Right-Censoring Applied! Parquet saved: {output_parquet}")
        print("\n📊 [DILIPLUS] Rigorous Cohort Label Distribution & Emulated Observation Windows:")
        print(df_stats.T.to_markdown())
        
    except Exception as e:
        print(f"🚨 [DILIPLUS] Label generation and censoring failed: {e}")

    print(f"\n⏱️ Total execution time: {time.time() - start_time:.2f}s")
    conn.close()

if __name__ == "__main__":
    extract_dili_labels_and_censor()

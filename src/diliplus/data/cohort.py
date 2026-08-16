"""
DILI-PLUS | 多重用药队列与肝功能时间线提取（包实现）

职责：筛选至少包含 5 种药物的住院记录，并对齐 ALT、AST、胆红素、
ALP、GGT 等肝功能化验。此阶段保留完整用药时间边界，不生成结局标签。
输入：外部医疗 DuckDB 中的分析视图和 laboratory_report_sub。
输出：data_cache/01_aligned_dili_labs.parquet 及队列化验汇总 CSV。
状态：当前 DILI 单任务数据流程的第一步。
安全：源数据库连接显式使用 read_only=True。
"""

import os
import pandas as pd
import time

from diliplus.config import load_settings
from diliplus.database import connect_source_database

def get_date_logic(col_name):
    return f"""
    COALESCE(
        TRY_CAST({col_name} AS TIMESTAMP),
        TRY_STRPTIME(CAST(TRY_CAST({col_name} AS BIGINT) AS VARCHAR), '%Y%m%d%H%M%S'),
        TRY_STRPTIME(CAST(TRY_CAST({col_name} AS BIGINT) AS VARCHAR), '%Y%m%d'),
        TRY_STRPTIME(CAST({col_name} AS VARCHAR), '%Y-%m-%d %H:%M:%S')
    )
    """

def build_dili_cohort(settings=None):
    settings = settings or load_settings()
    data_dir = str(settings.paths.data_cache)
    reports_dir = str(settings.paths.reports)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    db_path = str(settings.database_path)
    print(f"🔗 [DILIPLUS] Connecting to database: {db_path}")
    conn = connect_source_database(settings)
    start_time = time.time()

    print("\n⏳ [DILIPLUS] [1/2] Loading Polypharmacy Cohort Boundaries...")
    cohort_query = """
    CREATE OR REPLACE TEMP TABLE temp_cohort AS
    SELECT 
        encounter_id,
        patient_id AS health_reco,
        med_count AS unique_drugs,
        first_med_time,
        last_med_time
    FROM analysis.feature_medication_seq
    WHERE med_count >= 5
      AND first_med_time IS NOT NULL 
      AND last_med_time IS NOT NULL;
    """
    try:
        conn.execute(cohort_query)
        cohort_count = conn.execute("SELECT COUNT(*) FROM temp_cohort").fetchone()[0]
        print(f"✅ [DILIPLUS] Polypharmacy Cohort Ready: {cohort_count} high-risk encounters isolated.")
    except Exception as e:
        print(f"🚨 [DILIPLUS] Failed to read analysis views: {e}")
        return

    print("\n⏳ [DILIPLUS] [2/2] Extracting Comprehensive Hepatic Biomarkers...")
    
    # 扩展 DILI 相关标志物池
    aligned_labs_query = f"""
    CREATE OR REPLACE TEMP TABLE temp_aligned_labs AS
    WITH ValidLabs AS (
        SELECT 
            health_reco,
            TRIM(class_name) AS lab_item,
            result_valu AS lab_value,
            TRIM(abnormal_st) AS abnormal_status,
            {get_date_logic('effective_d')} AS lab_time
        FROM laboratory_report_sub
        WHERE (
            TRIM(class_name) LIKE '%谷丙转氨酶%' OR
            TRIM(class_name) LIKE '%谷草转氨酶%' OR
            TRIM(class_name) LIKE '%总胆红素%' OR
            TRIM(class_name) LIKE '%直接胆红素%' OR
            TRIM(class_name) LIKE '%碱性磷酸酶%' OR
            TRIM(class_name) LIKE '%谷氨酰%'
        )
        AND {get_date_logic('effective_d')} IS NOT NULL
    )
    SELECT 
        c.encounter_id,
        c.health_reco,
        c.first_med_time,
        c.last_med_time,
        c.unique_drugs,
        l.lab_item,
        l.lab_value,
        l.abnormal_status,
        l.lab_time
    FROM temp_cohort c
    INNER JOIN ValidLabs l 
        ON TRIM(CAST(c.health_reco AS VARCHAR)) = TRIM(CAST(l.health_reco AS VARCHAR))
    WHERE l.lab_time >= c.first_med_time 
      AND l.lab_time <= c.last_med_time + INTERVAL 7 DAY; -- 放宽至出院后 7 天以捕获迟发性肝损伤
    """
    
    try:
        conn.execute(aligned_labs_query)
        output_parquet = os.path.join(data_dir, "01_aligned_dili_labs.parquet")
        conn.execute(f"COPY temp_aligned_labs TO '{output_parquet}' (FORMAT PARQUET)")
        
        res_count = conn.execute("SELECT COUNT(*) FROM temp_aligned_labs").fetchone()[0]
        df_summary = conn.execute("""
            SELECT 
                lab_item, 
                COUNT(*) as test_count, 
                SUM(CASE WHEN abnormal_status IN ('H', '高', '↑', '↑↑', 'SH') THEN 1 ELSE 0 END) as abnormal_count
            FROM temp_aligned_labs 
            GROUP BY lab_item ORDER BY test_count DESC
        """).df()
        
        print(f"✅ [DILIPLUS] Hepatic Biomarkers Extracted: {res_count} valid lab events rescued.")
        print(f"📁 [DILIPLUS] Core tensor exported to: {output_parquet}")
        print("\n📊 [DILIPLUS] Rescued DILI Data Summary:")
        print(df_summary.to_markdown(index=False))
        df_summary.to_csv(os.path.join(reports_dir, "01_DILI_Cohort_Outcome_Summary.csv"), index=False)
        
    except Exception as e:
        print(f"🚨 [DILIPLUS] Lab linkage failed: {e}")

    print(f"\n⏱️ Total execution time: {time.time() - start_time:.2f}s")
    conn.close()

if __name__ == "__main__":
    build_dili_cohort()

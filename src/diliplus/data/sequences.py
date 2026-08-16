"""
DILI-PLUS | 用药—化验动态序列构建（包实现）

职责：按每次住院的 censor_time 截断事件，构建用药 Token/时间差序列与
化验项目/数值/时间差序列，并在 encounter_id 层面对齐两类动态模态。
输入：02_dili_labels_censored.parquet、01_aligned_dili_labs.parquet，
以及外部 DuckDB 的 analysis.feature_medications。
输出：data_cache/03_dili_dual_stream_tensors.parquet。
状态：当前 DILI 单任务的动态特征构建步骤。
安全：源数据库连接显式使用 read_only=True。
"""

import os
import pandas as pd
import numpy as np
import time

from diliplus.config import load_settings
from diliplus.database import connect_source_database

def build_dili_tensors(settings=None):
    settings = settings or load_settings()
    data_dir = str(settings.paths.data_cache)
    
    label_path = os.path.join(data_dir, "02_dili_labels_censored.parquet")
    lab_path = os.path.join(data_dir, "01_aligned_dili_labs.parquet")
    output_parquet = os.path.join(data_dir, "03_dili_dual_stream_tensors.parquet")
    
    print(f"🔗 [DILIPLUS] Connecting to DB & building Dual-Stream Tensors...")
    conn = connect_source_database(settings)
    start_time = time.time()

    # 🚀 将目标患者载入 Pandas 并反射到数据库，充当绝对拦截网
    print("⏳ [DILIPLUS] [0/3] Loading Target Cohort to RAM...")
    df_cohort = pd.read_parquet(label_path)[['encounter_id', 'censor_time', 'first_med_time', 'label_dili']]
    conn.execute("CREATE OR REPLACE TEMP TABLE mem_cohort AS SELECT encounter_id, censor_time, first_med_time FROM df_cohort")

    # ---------------------------------------------------------
    # 1. 构建药物动力学流 (Stream A) - 直取底层表，原生计算 Δt
    # ---------------------------------------------------------
    print("⏳ [DILIPLUS] [1/3] Extracting Stream A (Pharmacological) Direct from Source...")
    t_a = time.time()
    med_query = """
        WITH ValidMeds AS (
            SELECT 
                m.encounter_id,
                m.order_name as med_item,
                m.start_time as med_time,
                c.censor_time,
                c.first_med_time
            FROM analysis.feature_medications m
            INNER JOIN mem_cohort c ON m.encounter_id = c.encounter_id
            WHERE m.start_time <= c.censor_time -- The Guillotine (物理时间铡刀)
              AND m.order_name IS NOT NULL
        ),
        SortedMeds AS (
            SELECT 
                encounter_id,
                med_item,
                med_time,
                -- 计算连续物理时间差 \Delta t
                COALESCE(
                    EPOCH(med_time) - EPOCH(LAG(med_time) OVER(PARTITION BY encounter_id ORDER BY med_time)),
                    EPOCH(med_time) - EPOCH(first_med_time)
                ) / 3600.0 AS med_dt_hours
            FROM ValidMeds
        )
        SELECT 
            encounter_id,
            LIST(med_item ORDER BY med_time) as med_tokens,
            LIST(med_dt_hours ORDER BY med_time) as med_dt_hours
        FROM SortedMeds
        GROUP BY encounter_id
    """
    df_med = conn.execute(med_query).df()
    df_med['seq_length'] = df_med['med_tokens'].apply(len)
    print(f"   ✅ Processed Med stream in {time.time() - t_a:.2f}s. ({len(df_med)} encounters)")

    # ---------------------------------------------------------
    # 2. 构建生理反馈流 (Stream B)
    # ---------------------------------------------------------
    print("⏳ [DILIPLUS] [2/3] Reconstructing Stream B & Time-Decay Intervals...")
    t_b = time.time()
    lab_query = f"""
        WITH ValidLabs AS (
            SELECT 
                l.encounter_id,
                l.lab_item,
                TRY_CAST(l.lab_value AS FLOAT) as lab_value,
                l.lab_time,
                c.censor_time,
                c.first_med_time
            FROM read_parquet('{lab_path}') l
            INNER JOIN mem_cohort c ON l.encounter_id = c.encounter_id
            WHERE l.lab_time <= c.censor_time -- The Guillotine
              AND TRY_CAST(l.lab_value AS FLOAT) IS NOT NULL
        ),
        SortedLabs AS (
            SELECT 
                encounter_id,
                lab_item,
                lab_value,
                lab_time,
                -- 计算连续衰减时间差 \Delta t
                COALESCE(
                    EPOCH(lab_time) - EPOCH(LAG(lab_time) OVER(PARTITION BY encounter_id ORDER BY lab_time)),
                    EPOCH(lab_time) - EPOCH(first_med_time)
                ) / 3600.0 AS lab_dt_hours
            FROM ValidLabs
        )
        SELECT 
            encounter_id,
            LIST(lab_item ORDER BY lab_time) as lab_tokens,
            LIST(lab_value ORDER BY lab_time) as lab_values,
            LIST(lab_dt_hours ORDER BY lab_time) as lab_dt_hours
        FROM SortedLabs
        GROUP BY encounter_id
    """
    df_lab = conn.execute(lab_query).df()
    print(f"   ✅ Processed Lab stream in {time.time() - t_b:.2f}s. ({len(df_lab)} encounters)")
    
    # ---------------------------------------------------------
    # 3. 双流张量对齐与固化
    # ---------------------------------------------------------
    print("⏳ [DILIPLUS] [3/3] Assembling Dual-Stream Tensors...")
    # 内连接药物表确保包含用药记录，左连接化验表实现 Informative Missingness
    df_final = pd.merge(df_cohort, df_med, on='encounter_id', how='inner')
    df_final = pd.merge(df_final, df_lab, on='encounter_id', how='left')
    
    # 填充没有化验序列的患者
    for col in ['lab_tokens', 'lab_values', 'lab_dt_hours']:
        df_final[col] = df_final[col].apply(lambda x: x if isinstance(x, (np.ndarray, list)) else [])
    
    df_final.to_parquet(output_parquet)
    print(f"✅ [DILIPLUS] SUCCESS: {len(df_final)} encounters ready.")
    print(f"📁 [DILIPLUS] Tensor exported to: {output_parquet}")
    print(f"⏱️ Total Execution time: {time.time() - start_time:.2f}s")
    
    conn.close()

if __name__ == "__main__":
    build_dili_tensors()

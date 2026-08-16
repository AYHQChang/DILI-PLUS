"""
DILI-PLUS | 静态诊断特征构建（包实现）

职责：建立 encounter_id 与患者号的映射，汇总 ICD 诊断序列，并排除 K71 及
药物性/毒性肝病文字标签，降低直接目标编码泄露风险。
输入：03_dili_dual_stream_tensors.parquet 及外部 DuckDB 诊断相关表。
输出：data_cache/03b_diag_tensors.parquet。
状态：当前 DILI 单任务的静态诊断模态构建步骤。
安全：源数据库连接显式使用 read_only=True。
"""

import os
import pandas as pd
import time

from diliplus.config import load_settings
from diliplus.database import connect_source_database

def build_diag_tensors(settings=None):
    settings = settings or load_settings()
    data_dir = str(settings.paths.data_cache)
    
    cohort_path = os.path.join(data_dir, "03_dili_dual_stream_tensors.parquet")
    output_parquet = os.path.join(data_dir, "03b_diag_tensors.parquet")
    
    print(f"🔗 [DILIPLUS] Connecting to DB & building Diagnostic Tensors...")
    conn = connect_source_database(settings)
    start_time = time.time()

    print("⏳ [DILIPLUS] Extracting Diagnoses via Absolute Bridge (Expect < 3 seconds)...")
    diag_query = f"""
    WITH ValidCohort AS (
        SELECT encounter_id FROM read_parquet('{cohort_path}')
    ),
    -- 🌟 桥接核心：从最底层的明细表中，精准提取 encounter_id 与真实患者号的映射
    CohortBridge AS (
        SELECT DISTINCT 
            c.encounter_id, 
            TRIM(CAST(m.patient_id AS VARCHAR)) AS true_health_reco
        FROM analysis.feature_medications m
        INNER JOIN ValidCohort c ON m.encounter_id = c.encounter_id
    ),
    -- 🌟 极速哈希连接：用干净的患者号去撞击病案诊断表
    FilteredDiags AS (
        SELECT 
            b.encounter_id,
            d.diagnosis_c AS icd_code,  
            d.diagnosis_n AS diag_name  
        FROM in_medical_record_diag d 
        INNER JOIN CohortBridge b 
            ON TRIM(CAST(d.health_reco AS VARCHAR)) = b.true_health_reco
        WHERE d.diagnosis_c IS NOT NULL
          AND d.diagnosis_c NOT LIKE 'K71%' 
          AND d.diagnosis_n NOT LIKE '%药物性肝%' 
          AND d.diagnosis_n NOT LIKE '%毒性肝%'
    )
    SELECT 
        encounter_id,
        LIST(icd_code) as icd_codes,
        LIST(diag_name) as diag_names
    FROM FilteredDiags
    GROUP BY encounter_id
    """
    
    try:
        df_diag = conn.execute(diag_query).df()
        df_diag.to_parquet(output_parquet)
        print(f"✅ [DILIPLUS] Phenotype Tensors Built (Blinded K71): {len(df_diag)} encounters processed.")
        print(f"📁 [DILIPLUS] Tensor exported to: {output_parquet}")
    except Exception as e:
        print(f"🚨 [DILIPLUS] 诊断张量提取失败: {e}")

    print(f"⏱️ Total execution time: {time.time() - start_time:.2f}s")
    conn.close()

if __name__ == "__main__":
    build_diag_tensors()

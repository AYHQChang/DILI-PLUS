"""
DILI-PLUS | 当前队列基线特征表（包实现）

职责：以最终动态张量队列为样本基准，连接人口学、用药、合并症与入院附近化验，
分 DILI/非 DILI 组计算中位数、四分位数、比例、P 值和标准化差异。
输入：03_dili_dual_stream_tensors.parquet 与外部医疗 DuckDB。
输出：reports/Table_01_Baseline_Characteristics_DILIPLUS.txt。
状态：当前 DILI 单任务的 Table 1 生成脚本。
安全：源数据库连接显式使用 read_only=True。
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import logging
import warnings

from diliplus.config import load_settings
from diliplus.database import connect_source_database

warnings.filterwarnings('ignore')

def setup_logger(log_file_path):
    logger = logging.getLogger('DILI_Baseline_Stats')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def calc_smd_continuous(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0: return np.nan
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0: return 0
    return abs(np.mean(group1) - np.mean(group2)) / pooled_sd

def calc_smd_categorical(group1, group2):
    if len(group1) == 0 or len(group2) == 0: return np.nan
    p1 = np.mean(group1)
    p2 = np.mean(group2)
    pooled_sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    if pooled_sd == 0: return 0
    return abs(p1 - p2) / pooled_sd

def generate_table_1(settings=None):
    settings = settings or load_settings()
    data_dir = str(settings.model_data_dir)
    reports_dir = str(settings.paths.reports)
    os.makedirs(reports_dir, exist_ok=True)
    
    log_path = os.path.join(reports_dir, "Table_01_Baseline_Characteristics_DILIPLUS.txt")
    logger = setup_logger(log_path)
    
    # 🌟 核心改动：直接以 Stage 02 产生的带删失时间戳的 Labels 为唯一真理来源
    # labels_path = os.path.abspath(os.path.join(data_dir, "02_dili_labels_censored.parquet")).replace('\\', '/')
    # 直接读取最终送入模型的张量文件，确保 Table 1 与实验模型训练的 N 数 (51,316) 绝对一致
    labels_path = os.path.abspath(os.path.join(data_dir, "03_dili_dual_stream_tensors.parquet")).replace('\\', '/')
    
    if not os.path.exists(labels_path):
        logger.error(f"❌ Target Parquet not found: {labels_path}. Please run Stage 02 first.")
        return

    logger.info(f"🔗 [1/3] Hooking into DILIPLUS Cohort via Bounded Right-Censoring ...")
    
    # =========================================================================
    # 🚨 数据库 SQL 提取逻辑 (Data Extraction Logic)
    # =========================================================================
    # =========================================================================
    # 🚨 数据库 SQL 提取逻辑 (Data Extraction Logic) - 严格防止 JOIN 膨胀版
    # =========================================================================
    query = f"""
    WITH Cohort AS (
        SELECT DISTINCT
            encounter_id,
            label_dili,
            TRY_CAST(first_med_time AS TIMESTAMP) AS first_med_time,
            TRY_CAST(censor_time AS TIMESTAMP) AS censor_time
        FROM read_parquet('{labels_path}')
        WHERE label_dili IS NOT NULL
    ),
    Demographics AS (
        SELECT 
            e.encounter_id,
            MAX(e.patient_id) AS patient_id,
            MAX(date_diff('year', TRY_CAST(p.birth_date AS TIMESTAMP), TRY_CAST(e.admit_date AS TIMESTAMP))) AS age,
            MAX(TRY_CAST(e.admit_date AS TIMESTAMP)) AS admission_time,
            MAX(CASE WHEN UPPER(TRIM(CAST(p.gender AS VARCHAR))) IN ('1', '1.0', 'M', 'MALE', '男') THEN 1 ELSE 0 END) AS gender_male
        FROM analysis.v_patient_encounters e
        JOIN analysis.v_patient_profile p ON e.patient_id = p.patient_id
        INNER JOIN Cohort c ON e.encounter_id = c.encounter_id
        GROUP BY e.encounter_id  -- 强制聚合，彻底杜绝 1对多 造成的行数膨胀
    ),
    True_Med_Count AS (
        SELECT 
            m.encounter_id,
            COUNT(m.order_name) as actual_pre_index_meds
        FROM analysis.feature_medications m
        INNER JOIN Cohort c ON m.encounter_id = c.encounter_id
        WHERE m.start_time <= c.censor_time
        GROUP BY m.encounter_id
    ),
    Comorbidities AS (
        SELECT 
            d.encounter_id,
            MAX(CASE WHEN diag_code ILIKE 'E11%' THEN 1 ELSE 0 END) AS diabetes_flag,
            MAX(CASE WHEN diag_code ILIKE 'I10%' THEN 1 ELSE 0 END) AS hypertension_flag,
            MAX(CASE WHEN diag_code ILIKE 'I50%' THEN 1 ELSE 0 END) AS heart_failure_flag,
            MAX(CASE WHEN diag_code ILIKE 'A41%' OR diag_code ILIKE 'R57%' THEN 1 ELSE 0 END) AS sepsis_flag,
            MAX(CASE WHEN diag_code ILIKE 'K70%' OR diag_code ILIKE 'K74%' OR diag_code ILIKE 'K76%' THEN 1 ELSE 0 END) AS liver_disease_flag,
            MAX(CASE WHEN diag_code ILIKE 'B18%' THEN 1 ELSE 0 END) AS chronic_hepatitis_flag
        FROM analysis.feature_diagnoses d
        INNER JOIN Cohort c ON d.encounter_id = c.encounter_id
        GROUP BY d.encounter_id
    ),
    BaselineLab AS (
        SELECT 
            TRIM(CAST(e.encounter_id AS VARCHAR)) AS encounter_id,
            MEDIAN(CASE WHEN l.class_name LIKE '%丙氨酸氨基转移酶%' OR l.class_name LIKE '%谷丙%' THEN TRY_CAST(l.result_valu AS DOUBLE) END) AS baseline_alt,
            MEDIAN(CASE WHEN l.class_name LIKE '%天门冬氨酸氨基转移酶%' OR l.class_name LIKE '%谷草%' THEN TRY_CAST(l.result_valu AS DOUBLE) END) AS baseline_ast,
            MEDIAN(CASE WHEN l.class_name LIKE '%总胆红素%' THEN TRY_CAST(l.result_valu AS DOUBLE) END) AS baseline_tbil,
            MEDIAN(CASE WHEN l.class_name LIKE '%白蛋白%' THEN TRY_CAST(l.result_valu AS DOUBLE) END) AS baseline_albumin
        FROM Demographics e
        INNER JOIN laboratory_report_sub l ON TRIM(CAST(e.patient_id AS VARCHAR)) = TRIM(CAST(l.health_reco AS VARCHAR))
        WHERE COALESCE(
            TRY_CAST(l.effective_d AS TIMESTAMP),
            TRY_STRPTIME(CAST(TRY_CAST(l.effective_d AS BIGINT) AS VARCHAR), '%Y%m%d%H%M%S'),
            TRY_STRPTIME(CAST(TRY_CAST(l.effective_d AS BIGINT) AS VARCHAR), '%Y%m%d'),
            TRY_STRPTIME(CAST(l.effective_d AS VARCHAR), '%Y-%m-%d %H:%M:%S')
        ) BETWEEN e.admission_time - INTERVAL 2 DAY AND e.admission_time + INTERVAL 2 DAY
        GROUP BY 1
    )
    SELECT 
        c.label_dili,
        d.age,
        d.gender_male,
        -- 使用 GREATEST 防止时间差为 0 导致后续除法报错，同时保留这些有效行
        GREATEST(DATE_DIFF('hour', c.first_med_time, c.censor_time), 1) AS intervention_duration_hours,
        COALESCE(tm.actual_pre_index_meds, 0) / NULLIF(GREATEST(DATE_DIFF('hour', c.first_med_time, c.censor_time), 1) / 24.0, 0) AS med_density_per_day,
        cb.diabetes_flag,
        cb.hypertension_flag,
        cb.heart_failure_flag,
        cb.sepsis_flag,
        cb.liver_disease_flag,
        cb.chronic_hepatitis_flag,
        bl.baseline_alt,
        bl.baseline_ast,
        bl.baseline_tbil,
        bl.baseline_albumin
    FROM Cohort c
    LEFT JOIN Demographics d ON c.encounter_id = d.encounter_id
    LEFT JOIN True_Med_Count tm ON c.encounter_id = tm.encounter_id
    LEFT JOIN Comorbidities cb ON c.encounter_id = cb.encounter_id
    LEFT JOIN BaselineLab bl ON c.encounter_id = bl.encounter_id
    -- 🚨 删除了原有的 WHERE DATE_DIFF > 0，确保绝对 1:1 对齐 03 张量
    """
    
    try:
        conn = connect_source_database(settings)
        df = conn.execute(query).df()
        conn.close()
    except Exception as e:
        logger.error(f"❌ SQL Execution Failed: {e}")
        return

    logger.info(f"📊 [2/3] Total unified cohort loaded: {len(df)}. Commencing strict academic profiling...")

    # 更新特征字典名称以匹配新版的物理截断含义
    features = {
        'age': {'type': 'continuous', 'name': 'Age, years'},
        'gender_male': {'type': 'categorical', 'name': 'Sex, Male (%)'},
        'intervention_duration_hours': {'type': 'continuous', 'name': 'Observation Window (hours)'},
        'med_density_per_day': {'type': 'continuous', 'name': 'Medication Density (Events/24h)'},
        'sepsis_flag': {'type': 'categorical', 'name': 'Sepsis/Shock (ICD: A41, R57) (%)'},
        'heart_failure_flag': {'type': 'categorical', 'name': 'Heart Failure (ICD: I50) (%)'},
        'hypertension_flag': {'type': 'categorical', 'name': 'Hypertension (ICD: I10) (%)'},
        'diabetes_flag': {'type': 'categorical', 'name': 'Diabetes Type 2 (ICD: E11) (%)'},
        'liver_disease_flag': {'type': 'categorical', 'name': 'Baseline Liver Disease (K70, K74, K76) (%)'},
        'chronic_hepatitis_flag': {'type': 'categorical', 'name': 'Chronic Viral Hepatitis (ICD: B18) (%)'},
        'baseline_alt': {'type': 'continuous', 'name': 'Baseline ALT (U/L)'},
        'baseline_ast': {'type': 'continuous', 'name': 'Baseline AST (U/L)'},
        'baseline_tbil': {'type': 'continuous', 'name': 'Baseline TBIL (μmol/L)'},
        'baseline_albumin': {'type': 'continuous', 'name': 'Baseline Albumin (g/L)'}
    }

    available_features = {k: v for k, v in features.items() if k in df.columns and df[k].notna().sum() > 0}

    logger.info("=" * 140)
    logger.info(f"🏥 DILI-PLUS PROJECT - CLINICALLY RIGOROUS BASELINE REPORT (CENSORED)")
    logger.info("=" * 140)
    
    header = f"{'Characteristic':<45} | {'Total Cohort':<20} | {'Non-DILI':<20} | {'DILI':<20} | {'P-value':<10} | {'SMD':<8}"
    logger.info(header)
    logger.info("-" * 140)

    for col, f in available_features.items():
        group_total = df[col].dropna()
        group_non_dili = df[df['label_dili'] == 0][col].dropna()
        group_dili = df[df['label_dili'] == 1][col].dropna()
        
        if len(group_total) == 0: continue

        if f['type'] == 'continuous':
            def fmt_cont(s):
                if len(s) == 0: return "N/A"
                return f"{np.median(s):.1f} [{np.percentile(s, 25):.1f}-{np.percentile(s, 75):.1f}]"
            
            str_total = fmt_cont(group_total)
            str_non_dili = fmt_cont(group_non_dili)
            str_dili = fmt_cont(group_dili)
            
            stat, p_val = stats.mannwhitneyu(group_dili, group_non_dili, alternative='two-sided')
            smd = calc_smd_continuous(group_dili, group_non_dili)
                
        else:
            def fmt_cat(s):
                if len(s) == 0: return "N/A"
                count = s.sum()
                pct = (count / len(s)) * 100
                return f"{int(count)} ({pct:.1f}%)"
            
            str_total = fmt_cat(group_total)
            str_non_dili = fmt_cat(group_non_dili)
            str_dili = fmt_cat(group_dili)
            
            contingency_table = pd.crosstab(df['label_dili'], df[col])
            if contingency_table.shape == (2, 2):
                chi2, p_val, dof, ex = stats.chi2_contingency(contingency_table)
            else:
                p_val = np.nan
            smd = calc_smd_categorical(group_dili, group_non_dili)

        p_str = "<0.001" if pd.notna(p_val) and p_val < 0.001 else (f"{p_val:.3f}" if pd.notna(p_val) else "N/A")
        smd_str = f"{smd:.3f}" if pd.notna(smd) else "N/A"
        
        row_str = f"{f['name']:<45} | {str_total:<20} | {str_non_dili:<20} | {str_dili:<20} | {p_str:<10} | {smd_str:<8}"
        logger.info(row_str)

    logger.info("=" * 140)
    logger.info("Note: Continuous variables presented as Median [Q1-Q3]. Categorical as n (%).")
    logger.info("SMD = Standardised Mean Difference. SMD > 0.100 generally indicates a meaningful imbalance.")
    logger.info(f"✅ [3/3] Baseline extraction complete. Report saved to {log_path}")

if __name__ == "__main__":
    generate_table_1()

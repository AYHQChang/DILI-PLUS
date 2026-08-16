"""
DILI-PLUS | 基线特征表旧版备份（已归档）

职责：使用较早的自适应 schema 探测与基线窗口逻辑生成 DILI 队列表 1。
输入：外部医疗 DuckDB 与标签 Parquet。
输出：历史版基线特征统计报告。
状态：备份脚本，不属于当前主流程；现行版本为 Table_01_baseline_characteristics_DILI.py。
安全：源数据库连接显式使用 read_only=True。
"""

import os
import duckdb
import pandas as pd
import numpy as np
from scipy import stats
import logging
import warnings

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

def generate_table_1():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data_cache")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    log_path = os.path.join(reports_dir, "Table_01_Baseline_Characteristics_DILI_v4.txt")
    logger = setup_logger(log_path)
    
    db_path = r"D:\MedicalAI_Work\duck\medical.duckdb"
    labels_path = os.path.abspath(os.path.join(data_dir, "02_ade_labels.parquet")).replace('\\', '/')
    
    logger.info(f"🔗 [1/4] Connecting to DuckDB to dynamically map schema...")
    
    try:
        conn = duckdb.connect(db_path, read_only=True)
        # 🌟 动态探测 feature_medications 表的列名，防止写死导致报错
        med_cols = conn.execute("DESCRIBE analysis.feature_medications").df()['column_name'].tolist()
        
        # 寻找药品名称列
        drug_col = 'medicine_name' # default
        for c in ['atc_code', 'medicine_na', 'medicine_name', 'cn_medicine', 'drug_name', 'item_name', 'drug_standa']:
            if c in med_cols:
                drug_col = c
                break
                
        # 寻找时间列
        time_col = 'order_time'
        for c in ['order_time', 'start_time', 'sqctime', 'create_time']:
            if c in med_cols:
                time_col = c
                break
                
        logger.info(f"   ✅ Discovered mapping -> Drug Column: [{drug_col}], Time Column: [{time_col}]")
    except Exception as e:
        logger.error(f"❌ Failed to connect to duckdb for schema probing: {e}")
        return

    logger.info(f"🔗 [2/4] Executing Causal-Strict Query...")
    
    query = f"""
    WITH Labelled AS (
        SELECT 
            encounter_id, 
            label_dili,
            TRY_CAST(dili_event_time AS TIMESTAMP) as event_time
        FROM read_parquet('{labels_path}') 
        WHERE label_dili IS NOT NULL
    ),
    Demographics AS (
        SELECT 
            e.encounter_id,
            e.patient_id,
            date_diff('year', p.birth_date, e.admit_date) AS age,
            TRY_CAST(e.admit_date AS TIMESTAMP) AS admission_time,
            CASE WHEN UPPER(TRIM(CAST(p.gender AS VARCHAR))) IN ('1', '1.0', 'M', 'MALE', '男') THEN 1 ELSE 0 END AS gender_male
        FROM analysis.v_patient_encounters e
        JOIN analysis.v_patient_profile p ON e.patient_id = p.patient_id
    ),
    -- 🌟 阻断目标泄露：在明细表层级切断发病后用药
    MedSeq_Censored AS (
        SELECT 
            m.encounter_id,
            MIN(TRY_CAST(m.{time_col} AS TIMESTAMP)) AS first_med_time,
            COUNT(DISTINCT m.{drug_col}) AS unique_drugs_censored
        FROM analysis.feature_medications m
        JOIN Labelled l ON m.encounter_id = l.encounter_id
        WHERE m.{drug_col} IS NOT NULL 
          AND (l.event_time IS NULL OR TRY_CAST(m.{time_col} AS TIMESTAMP) < l.event_time)
        GROUP BY m.encounter_id
    ),
    Comorbidities AS (
        SELECT 
            encounter_id,
            MAX(CASE WHEN diag_code ILIKE 'E11%' THEN 1 ELSE 0 END) AS diabetes_flag,
            MAX(CASE WHEN diag_code ILIKE 'I10%' THEN 1 ELSE 0 END) AS hypertension_flag,
            MAX(CASE WHEN diag_code ILIKE 'I50%' THEN 1 ELSE 0 END) AS heart_failure_flag,
            MAX(CASE WHEN diag_code ILIKE 'A41%' OR diag_code ILIKE 'R57%' THEN 1 ELSE 0 END) AS sepsis_flag,
            MAX(CASE WHEN diag_code ILIKE 'K70%' OR diag_code ILIKE 'K74%' OR diag_code ILIKE 'K76%' THEN 1 ELSE 0 END) AS liver_disease_flag,
            MAX(CASE WHEN diag_code ILIKE 'B18%' THEN 1 ELSE 0 END) AS chronic_hepatitis_flag
        FROM analysis.feature_diagnoses
        GROUP BY encounter_id
    ),
    -- 🌟 激活基线检验：利用 ROW_NUMBER 提取入院 -7d 到 +48h 内的第一份化验单
    BaselineLab AS (
        SELECT 
            encounter_id,
            MEDIAN(CASE WHEN class_name LIKE '%丙氨酸氨基转移酶%' OR class_name LIKE '%谷丙%' THEN result_valu END) AS baseline_alt,
            MEDIAN(CASE WHEN class_name LIKE '%天门冬氨酸氨基转移酶%' OR class_name LIKE '%谷草%' THEN result_valu END) AS baseline_ast,
            MEDIAN(CASE WHEN class_name LIKE '%总胆红素%' THEN result_valu END) AS baseline_tbil,
            MEDIAN(CASE WHEN class_name LIKE '%白蛋白%' THEN result_valu END) AS baseline_albumin
        FROM (
            SELECT 
                TRIM(CAST(e.encounter_id AS VARCHAR)) AS encounter_id,
                l.class_name,
                TRY_CAST(l.result_valu AS DOUBLE) AS result_valu,
                ROW_NUMBER() OVER(PARTITION BY e.encounter_id, l.class_name ORDER BY TRY_CAST(l.effective_d AS TIMESTAMP) ASC) as rn
            FROM Demographics e
            INNER JOIN laboratory_report_sub l ON TRIM(CAST(e.patient_id AS VARCHAR)) = TRIM(CAST(l.health_reco AS VARCHAR))
            WHERE TRY_CAST(l.effective_d AS TIMESTAMP) BETWEEN e.admission_time - INTERVAL 7 DAY AND e.admission_time + INTERVAL 48 HOUR
        )
        WHERE rn = 1
        GROUP BY 1
    )
    SELECT 
        l.label_dili,
        m.unique_drugs_censored AS unique_drugs,
        d.age,
        d.gender_male,
        DATE_DIFF('hour', d.admission_time, m.first_med_time) AS time_to_initial_medication_hours,
        c.diabetes_flag,
        c.hypertension_flag,
        c.heart_failure_flag,
        c.sepsis_flag,
        c.liver_disease_flag,
        c.chronic_hepatitis_flag,
        b.baseline_alt,
        b.baseline_ast,
        b.baseline_tbil,
        b.baseline_albumin
    FROM Labelled l
    LEFT JOIN MedSeq_Censored m ON l.encounter_id = m.encounter_id
    LEFT JOIN Demographics d ON l.encounter_id = d.encounter_id
    LEFT JOIN Comorbidities c ON l.encounter_id = c.encounter_id
    LEFT JOIN BaselineLab b ON l.encounter_id = b.encounter_id
    WHERE DATE_DIFF('hour', d.admission_time, m.first_med_time) >= 0
    """
    
    try:
        df = conn.execute(query).df()
        conn.close()
    except Exception as e:
        logger.error(f"❌ SQL Execution Failed: {e}")
        return

    logger.info(f"📊 [3/4] Total cohort loaded: {len(df)}. Commencing strict academic profiling...")

    features = {
        'age': {'type': 'continuous', 'name': 'Age, years'},
        'gender_male': {'type': 'categorical', 'name': 'Sex, Male (%)'},
        'time_to_initial_medication_hours': {'type': 'continuous', 'name': 'Time to Initial Medication (hours)'},
        'hypertension_flag': {'type': 'categorical', 'name': 'Hypertension (I10) (%)'},
        'diabetes_flag': {'type': 'categorical', 'name': 'Diabetes Type 2 (E11) (%)'},
        'liver_disease_flag': {'type': 'categorical', 'name': 'Baseline Liver Disease (K70, K74, K76) (%)'},
        'chronic_hepatitis_flag': {'type': 'categorical', 'name': 'Chronic Viral Hepatitis (B18) (%)'},
        'heart_failure_flag': {'type': 'categorical', 'name': 'Heart Failure (I50) (%)'},
        'sepsis_flag': {'type': 'categorical', 'name': 'Sepsis/Shock (A41, R57) (%)'},
        'unique_drugs': {'type': 'continuous', 'name': 'Strict Pre-DILI Medication Classes (count)'},
        'baseline_alt': {'type': 'continuous', 'name': 'Baseline ALT (U/L)'},
        'baseline_ast': {'type': 'continuous', 'name': 'Baseline AST (U/L)'},
        'baseline_tbil': {'type': 'continuous', 'name': 'Baseline TBIL (μmol/L)'},
        'baseline_albumin': {'type': 'continuous', 'name': 'Baseline Albumin (g/L)'}
    }

    available_features = {k: v for k, v in features.items() if k in df.columns and df[k].notna().sum() > 0}

    logger.info("=" * 140)
    logger.info(f"🏥 DILI PREDICTION PROJECT - CLINICALLY RIGOROUS BASELINE REPORT")
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
    logger.info("SMD = Standardised Mean Difference. SMD > 0.1 generally indicates a meaningful imbalance.")
    logger.info(f"✅ [4/4] Baseline extraction complete. Report saved to {log_path}")

if __name__ == "__main__":
    generate_table_1()

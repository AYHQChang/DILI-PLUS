"""
DILI-PLUS | 未校准传统机器学习基线（包实现）

职责：将用药、化验项目和诊断序列转换为折内拟合的 TF-IDF 特征，使用五折
GroupKFold 评估 Logistic Regression 与 XGBoost，并保存逐样本预测。
输入：03_dili_dual_stream_tensors.parquet、03b_diag_tensors.parquet。
输出：未校准模型权重、reports/predictions/ 和 05_Experiment_Results_Table.csv。
状态：用于未校准对照与校准影响分析；最终校准实验使用对应 calibrated 脚本。
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from scipy.sparse import hstack
import warnings
warnings.filterwarnings("ignore")

from diliplus.config import load_settings

# ---------------------------------------------------------
# 🌟 高阶临床指标与置信区间计算 (保持与 DL 脚本一致)
# ---------------------------------------------------------
def calculate_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        bin_idx = binids == i
        if np.sum(bin_idx) > 0:
            prob_mean = np.mean(y_prob[bin_idx])
            acc_mean = np.mean(y_true[bin_idx])
            ece += (np.sum(bin_idx) / len(y_prob)) * np.abs(prob_mean - acc_mean)
    return ece

def calculate_sci_metrics_with_ci(y_true, y_prob, n_bootstraps=1000):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = calculate_ece(y_true, y_prob)
    
    bootstrapped_auroc = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2: continue
        bootstrapped_auroc.append(roc_auc_score(y_true[indices], y_prob[indices]))
        
    ci_lower = np.percentile(bootstrapped_auroc, 2.5) if bootstrapped_auroc else auroc
    ci_upper = np.percentile(bootstrapped_auroc, 97.5) if bootstrapped_auroc else auroc

    return {
        "AUROC": auroc,
        "AUROC_95CI_Lower": ci_lower,
        "AUROC_95CI_Upper": ci_upper,
        "AUPRC": auprc,
        "Brier": brier,
        "ECE": ece
    }

# ---------------------------------------------------------
# 🌟 数据加载与特征展平
# ---------------------------------------------------------
def load_and_flatten_data(data_dir):
    print("⏳ [ML Baseline] Loading DILIPLUS Tensors and Flattening to TF-IDF corpus...")
    df_med_lab = pd.read_parquet(os.path.join(data_dir, "03_dili_dual_stream_tensors.parquet"))
    df_diag = pd.read_parquet(os.path.join(data_dir, "03b_diag_tensors.parquet"))
    df = pd.merge(df_med_lab, df_diag, on='encounter_id', how='left')
    
    df['health_reco'] = df['encounter_id'].astype(str).apply(lambda x: x.split('_')[0])
    
    def to_string(x):
        if isinstance(x, (list, np.ndarray)): return " ".join([str(i) for i in x])
        return ""
        
    df['med_str'] = df['med_tokens'].apply(to_string)
    df['lab_str'] = df['lab_tokens'].apply(to_string)
    df['diag_str'] = df['icd_codes'].apply(to_string)
    
    return df

# ---------------------------------------------------------
# 🌟 训练主循环
# ---------------------------------------------------------
def train_ml_baselines(settings=None):
    settings = settings or load_settings()
    data_dir = str(settings.paths.data_cache)
    save_dir = str(settings.paths.checkpoints)
    report_dir = str(settings.paths.reports)
    
    # 🔥 建立预测概率存储专区
    preds_dir = os.path.join(report_dir, "predictions")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    
    df = load_and_flatten_data(data_dir)
    groups = df['health_reco'].values
    y = df['label_dili'].values
    
    models_to_train = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric='logloss')
    }
    
    all_fold_results = []
    gkf = GroupKFold(n_splits=5)
    
    print(f"🚀 [ML Baseline] Starting 5-Fold Evaluation with Patient-Level Isolation...")
    for model_name, model_obj in models_to_train.items():
        print(f"\n{'='*50}\nEvaluating ML Model: {model_name}\n{'='*50}")
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(df, y, groups)):
            vec_med = TfidfVectorizer(max_features=2000)
            vec_lab = TfidfVectorizer(max_features=500)
            vec_diag = TfidfVectorizer(max_features=1000)
            
            X_train_med = vec_med.fit_transform(df.iloc[train_idx]['med_str'])
            X_train_lab = vec_lab.fit_transform(df.iloc[train_idx]['lab_str'])
            X_train_diag = vec_diag.fit_transform(df.iloc[train_idx]['diag_str'])
            X_train = hstack([X_train_med, X_train_lab, X_train_diag])
            y_train = y[train_idx]
            
            X_test_med = vec_med.transform(df.iloc[test_idx]['med_str'])
            X_test_lab = vec_lab.transform(df.iloc[test_idx]['lab_str'])
            X_test_diag = vec_diag.transform(df.iloc[test_idx]['diag_str'])
            X_test = hstack([X_test_med, X_test_lab, X_test_diag])
            y_test = y[test_idx]
            
            model = model_obj
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # 🔥 落地保存本折的 y_true 和 y_prob 供后期画图
            pred_df = pd.DataFrame({'y_true': y_test, 'y_prob': y_prob, 'fold': fold+1})
            pred_df.to_csv(os.path.join(preds_dir, f"preds_{model_name}_Fold{fold+1}.csv"), index=False)
            
            metrics = calculate_sci_metrics_with_ci(y_test, y_prob, n_bootstraps=1000)
            metrics.update({"Model_Architecture": model_name, "Fold": fold + 1})
            all_fold_results.append(metrics)
            
            print(f"   Fold {fold+1} | AUROC: {metrics['AUROC']:.4f} (95% CI: {metrics['AUROC_95CI_Lower']:.4f}-{metrics['AUROC_95CI_Upper']:.4f}) | Brier: {metrics['Brier']:.4f}")
            joblib.dump(model, os.path.join(save_dir, f"best_{model_name}_Fold{fold+1}.joblib"))

    report_path = os.path.join(report_dir, "05_Experiment_Results_Table.csv")
    df_results = pd.DataFrame(all_fold_results)
    if os.path.exists(report_path):
        df_existing = pd.read_csv(report_path)
        df_results = pd.concat([df_existing, df_results], ignore_index=True)
    df_results.to_csv(report_path, index=False)
    print(f"\n✅ ML Baselines Results appended to {report_path}")

if __name__ == "__main__":
    train_ml_baselines()

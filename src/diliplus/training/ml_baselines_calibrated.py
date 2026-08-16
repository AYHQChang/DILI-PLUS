"""
DILI-PLUS | 温度缩放后的传统机器学习基线（包实现）

职责：以 TF-IDF 表示训练 Logistic Regression 与 XGBoost；每个 GroupKFold 外层折
内部再划分训练集和温度校准集，并仅在外层测试集报告最终指标。
输入：03_dili_dual_stream_tensors.parquet、03b_diag_tensors.parquet。
输出：reports/predictions_calibrated/ 和 05_Calibrated_Results_Table.csv。
状态：当前 DILI 单任务的传统机器学习主评估路径。
说明：输出表采用追加写入，重复运行前需由调用方管理历史结果。
"""

import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from scipy.sparse import hstack
import warnings
warnings.filterwarnings("ignore")

from diliplus.config import load_settings

# =============================================================================
# 🌟 第一部分：顶刊级评估指标 (与 DL 绝对对齐)
# =============================================================================
def calculate_partial_auc(y_true, y_prob, fpr_limit=0.2):
    """计算 FPR <= 0.2 区间内的 Partial AUC"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    if fpr_limit <= 0 or fpr_limit > 1: return 0.0
    valid_idx = np.where(fpr <= fpr_limit)[0]
    if len(valid_idx) < 2: return 0.0
    
    fpr_part = fpr[valid_idx]
    tpr_part = tpr[valid_idx]
    
    # 线性插值闭合边界
    if fpr_part[-1] < fpr_limit:
        idx_next = valid_idx[-1] + 1
        if idx_next < len(fpr):
            slope = (tpr[idx_next] - tpr_part[-1]) / (fpr[idx_next] - fpr_part[-1] + 1e-9)
            tpr_interp = tpr_part[-1] + slope * (fpr_limit - fpr_part[-1])
            fpr_part = np.append(fpr_part, fpr_limit)
            tpr_part = np.append(tpr_part, tpr_interp)
            
    pauc = np.trapz(tpr_part, fpr_part)
    return pauc / (fpr_limit * 1.0)

def calculate_net_benefit(y_true, y_prob, thresholds=np.arange(0.01, 1.0, 0.01)):
    """计算临床决策曲线 (Decision Curve Analysis, DCA) 的 Net Benefit 及 AUDC"""
    net_benefits = []
    n = len(y_true)
    if n == 0: return 0.0
    for pt in thresholds:
        preds = (y_prob >= pt).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits.append(nb)
    return np.trapz(np.maximum(net_benefits, 0), thresholds)

def calculate_quantile_ece(y_true, y_prob, n_bins=10):
    """等频分箱的校准误差 (Quantile ECE)"""
    try:
        bins = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        bins[-1] += 1e-8 
        binids = np.digitize(y_prob, bins) - 1
    except:
        bins = np.linspace(0, 1, n_bins + 1)
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
    """汇总所有指标并计算 Bootstrap 95% 置信区间 (完全对齐 DL 返回的 Keys)"""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = calculate_quantile_ece(y_true, y_prob)
    pauc = calculate_partial_auc(y_true, y_prob)
    audc = calculate_net_benefit(y_true, y_prob)
    
    rng = np.random.RandomState(42)
    bootstrapped_auroc = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2: continue
        bootstrapped_auroc.append(roc_auc_score(y_true[indices], y_prob[indices]))
        
    ci_lower = np.percentile(bootstrapped_auroc, 2.5) if bootstrapped_auroc else auroc
    ci_upper = np.percentile(bootstrapped_auroc, 97.5) if bootstrapped_auroc else auroc

    return {
        "AUROC": auroc, "AUROC_95CI_Lower": ci_lower, "AUROC_95CI_Upper": ci_upper,
        "AUPRC": auprc, "Brier": brier, "Quantile_ECE": ece, "pAUC_0.2": pauc, "NetBenefit_AUDC": audc
    }

# =============================================================================
# 🌟 第二部分：跨界温度缩放器 (ML Probabilities -> Logits -> L-BFGS -> Scaled Probabilities)
# =============================================================================
class TemperatureScaler(nn.Module):
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

def fit_temperature_scaling(val_probas, val_labels):
    """提取 sklearn 输出的 [N, 2] 概率，利用 PyTorch L-BFGS 寻找最优 T"""
    eps = 1e-9
    val_probas = np.clip(val_probas, eps, 1.0 - eps)
    val_logits = np.log(val_probas) # 逆向映射得到 [N, 2] 伪 logits
    
    logits_tensor = torch.tensor(val_logits, dtype=torch.float32)
    labels_tensor = torch.tensor(val_labels, dtype=torch.long)
    
    scaler = TemperatureScaler()
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    
    def eval_closure():
        optimizer.zero_grad()
        loss = nll_criterion(scaler(logits_tensor), labels_tensor)
        loss.backward()
        return loss
    
    optimizer.step(eval_closure)
    best_T = scaler.temperature.item()
    print(f"      🌡️ L-BFGS Optimized Temperature T = {best_T:.4f}")
    return best_T

def apply_temperature_scaling(test_probas, T):
    """使用最优温度 T 缩放测试集概率"""
    eps = 1e-9
    test_probas = np.clip(test_probas, eps, 1.0 - eps)
    test_logits = np.log(test_probas)
    
    scaled_logits = torch.tensor(test_logits, dtype=torch.float32) / T
    calibrated_probas = torch.softmax(scaled_logits, dim=1).numpy()
    
    # 返回阳性类的预测概率 [:, 1]
    return calibrated_probas[:, 1]

# =============================================================================
# 🌟 第三部分：数据加载与主控循环
# =============================================================================
def load_and_flatten_data(data_dir):
    print("⏳ [ML Baseline] Loading DILIPLUS Tensors and Flattening to TF-IDF corpus...")
    df_med_lab = pd.read_parquet(os.path.join(data_dir, "03_dili_dual_stream_tensors.parquet"))
    df_diag = pd.read_parquet(os.path.join(data_dir, "03b_diag_tensors.parquet"))
    df = pd.merge(df_med_lab, df_diag, on='encounter_id', how='left')
    
    # 提取用于 GroupKFold 隔离的标识
    df['health_reco'] = df['encounter_id'].astype(str).apply(lambda x: x.split('_')[0])
    
    def to_string(x):
        return " ".join([str(i) for i in x]) if isinstance(x, (list, np.ndarray)) else ""
        
    df['med_str'] = df['med_tokens'].apply(to_string)
    df['lab_str'] = df['lab_tokens'].apply(to_string)
    df['diag_str'] = df['icd_codes'].apply(to_string)
    return df

def main(settings=None):
    settings = settings or load_settings()
    data_dir = str(settings.paths.data_cache)
    save_dir = str(settings.paths.checkpoints)
    report_dir = str(settings.paths.reports)
    preds_dir = os.path.join(report_dir, "predictions_calibrated")
    
    os.makedirs(save_dir, exist_ok=True)
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
    
    print(f"🚀 [ML Baseline] Starting 5-Fold Evaluation with Temperature Scaling...")
    for model_name, model_obj in models_to_train.items():
        print(f"\n{'='*50}\nEvaluating Model: {model_name}\n{'='*50}")
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(df, y, groups)):
            
            # 🔥 与 DL 严格对齐：切出 15% 内部验证集用于拟合温度 T，杜绝数据泄露
            train_groups = groups[train_idx]
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
            train_sub_loc, val_loc = next(gss.split(train_idx, groups=train_groups))
            train_sub_idx, val_idx = train_idx[train_sub_loc], train_idx[val_loc]
            
            vec_med = TfidfVectorizer(max_features=2000)
            vec_lab = TfidfVectorizer(max_features=500)
            vec_diag = TfidfVectorizer(max_features=1000)
            
            # 拟合并转换训练主干集
            X_train_sub = hstack([
                vec_med.fit_transform(df.iloc[train_sub_idx]['med_str']),
                vec_lab.fit_transform(df.iloc[train_sub_idx]['lab_str']),
                vec_diag.fit_transform(df.iloc[train_sub_idx]['diag_str'])
            ])
            y_train_sub = y[train_sub_idx]
            
            # 仅转换验证集
            X_val = hstack([
                vec_med.transform(df.iloc[val_idx]['med_str']),
                vec_lab.transform(df.iloc[val_idx]['lab_str']),
                vec_diag.transform(df.iloc[val_idx]['diag_str'])
            ])
            y_val = y[val_idx]
            
            # 仅转换测试集
            X_test = hstack([
                vec_med.transform(df.iloc[test_idx]['med_str']),
                vec_lab.transform(df.iloc[test_idx]['lab_str']),
                vec_diag.transform(df.iloc[test_idx]['diag_str'])
            ])
            y_test = y[test_idx]
            
            # 1. 训练基线模型
            model = model_obj
            model.fit(X_train_sub, y_train_sub)
            
            # 2. 在验证集上提取概率并寻找最佳温度 T
            val_probas = model.predict_proba(X_val) # 输出形状 [N, 2]
            best_T = fit_temperature_scaling(val_probas, y_val)
            
            # 3. 在测试集上进行预测，并应用最优温度进行平滑校准
            test_probas_raw = model.predict_proba(X_test)
            test_preds_calib = apply_temperature_scaling(test_probas_raw, best_T)
            
            # 4. 计算大满贯指标并更新字典
            metrics = calculate_sci_metrics_with_ci(y_test, test_preds_calib, n_bootstraps=1000)
            metrics.update({"Model_Architecture": model_name + "_Calibrated", "Fold": fold + 1})
            all_fold_results.append(metrics)
            
            print(f"   ✅ Fold {fold+1} | AUROC: {metrics['AUROC']:.4f} | Quantile ECE: {metrics['Quantile_ECE']:.4f} | pAUC: {metrics['pAUC_0.2']:.4f}")
            
            # 5. 持久化校准后的测试集微观概率，用于后期画阴影图
            pred_df = pd.DataFrame({'y_true': y_test, 'y_prob': test_preds_calib, 'fold': fold+1})
            pred_df.to_csv(os.path.join(preds_dir, f"preds_calib_{model_name}_Fold{fold+1}.csv"), index=False)

    # =============================================================================
    # 🌟 统一追加汇入主表！(使用 pd.concat 追加模式，绝不覆盖)
    # =============================================================================
    report_path = os.path.join(report_dir, "05_Calibrated_Results_Table.csv")
    df_results = pd.DataFrame(all_fold_results)
    
    if os.path.exists(report_path):
        df_existing = pd.read_csv(report_path)
        # 将本次 ML 产生的 10 行数据追加到 DL 产生的 20 行数据下方
        df_results = pd.concat([df_existing, df_results], ignore_index=True)
        
    df_results.to_csv(report_path, index=False)
    print(f"\n✅ Calibrated ML Baselines successfully appended to {report_path}")

if __name__ == "__main__":
    main()

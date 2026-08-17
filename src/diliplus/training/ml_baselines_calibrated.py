"""
DILI-PLUS | 温度缩放后的传统机器学习基线（包实现）

职责：以 TF-IDF 表示训练 Logistic Regression 与 XGBoost；使用与深度模型相同的
training/selection/calibration/test 四方 grouped split，并仅对 outer test 推理一次。
输入：03_dili_dual_stream_tensors.parquet、03b_diag_tensors.parquet。
输出：run-specific 配对 raw/calibrated 预测、指标和版本化 sklearn artifact。
状态：当前 DILI 单任务的传统机器学习主评估路径。
说明：固定超参数基线不读取 selection partition；该分区仍保留以维持统一协议。
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.base import clone
from scipy.sparse import hstack
import warnings
warnings.filterwarnings("ignore")

from diliplus.config import load_settings
from diliplus.reproducibility import DEFAULT_SEED, derive_seed, seed_everything
from diliplus.artifacts import (
    build_artifact_metadata,
    config_snapshot,
    dataset_fingerprint,
    run_report_dir,
    save_sklearn_artifact,
    sklearn_artifact_path,
)
from diliplus.calibration import (
    fit_temperature,
    probabilities_from_logits,
    probabilities_to_logits,
)
from diliplus.splits import build_nested_grouped_splits

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

def calculate_sci_metrics_with_ci(
    y_true, y_prob, n_bootstraps=1000, bootstrap_seed=DEFAULT_SEED
):
    """汇总所有指标并计算 Bootstrap 95% 置信区间 (完全对齐 DL 返回的 Keys)"""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = calculate_quantile_ece(y_true, y_prob)
    pauc = calculate_partial_auc(y_true, y_prob)
    audc = calculate_net_benefit(y_true, y_prob)
    
    rng = np.random.default_rng(bootstrap_seed)
    bootstrapped_auroc = []
    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2: continue
        bootstrapped_auroc.append(roc_auc_score(y_true[indices], y_prob[indices]))
        
    ci_lower = np.percentile(bootstrapped_auroc, 2.5) if bootstrapped_auroc else auroc
    ci_upper = np.percentile(bootstrapped_auroc, 97.5) if bootstrapped_auroc else auroc

    return {
        "AUROC": auroc, "AUROC_95CI_Lower": ci_lower, "AUROC_95CI_Upper": ci_upper,
        "AUPRC": auprc, "Brier": brier, "Quantile_ECE": ece, "pAUC_0.2": pauc, "NetBenefit_AUDC": audc
    }

# =============================================================================
# 🌟 第三部分：数据加载与主控循环
# =============================================================================
def load_and_flatten_data(data_dir):
    print("⏳ [ML Baseline] Loading DILIPLUS Tensors and Flattening to TF-IDF corpus...")
    df_med_lab = pd.read_parquet(os.path.join(data_dir, "03_dili_dual_stream_tensors.parquet"))
    df_diag = pd.read_parquet(os.path.join(data_dir, "03b_diag_tensors.parquet"))
    df = pd.merge(df_med_lab, df_diag, on='encounter_id', how='left')
    
    def to_string(x):
        return " ".join([str(i) for i in x]) if isinstance(x, (list, np.ndarray)) else ""
        
    df['med_str'] = df['med_tokens'].apply(to_string)
    df['lab_str'] = df['lab_tokens'].apply(to_string)
    df['diag_str'] = df['icd_codes'].apply(to_string)
    return df

def main(argv=None, settings=None):
    settings = settings or load_settings()
    seed_everything(settings.reproducibility)
    parser = argparse.ArgumentParser(description="DILI-PLUS grouped calibrated ML baselines")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    data_dir = str(settings.model_data_dir)
    report_root = run_report_dir(settings, args.run_id)
    preds_dir = report_root / "predictions"
    metrics_dir = report_root / "metrics"
    preds_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_and_flatten_data(data_dir)
    y = df['label_dili'].values
    
    models_to_train = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=settings.reproducibility.global_seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=settings.reproducibility.global_seed,
            n_jobs=1,
        )
    }
    
    encounter_ids = df["encounter_id"].astype(str).to_numpy()
    labels = np.asarray(y, dtype=np.int64)
    folds = build_nested_grouped_splits(encounter_ids, labels, settings)
    data_fingerprint = dataset_fingerprint(settings)
    all_fold_results = []
    
    print(f"🚀 [ML Baseline] Starting 5-Fold Evaluation with Temperature Scaling...")
    for model_name, model_obj in models_to_train.items():
        print(f"\n{'='*50}\nEvaluating Model: {model_name}\n{'='*50}")
        
        model_config = config_snapshot(
            settings,
            {
                "model": model_name,
                "representation": "TF-IDF",
                "estimator_parameters": model_obj.get_params(deep=False),
                "selection_partition_usage": "reserved_not_used_fixed_hyperparameters",
            },
        )
        for split in folds:
            fold = split.fold
            
            vec_med = TfidfVectorizer(max_features=2000)
            vec_lab = TfidfVectorizer(max_features=500)
            vec_diag = TfidfVectorizer(max_features=1000)
            
            # 拟合并转换训练主干集
            X_train_sub = hstack([
                vec_med.fit_transform(df.iloc[split.training]['med_str']),
                vec_lab.fit_transform(df.iloc[split.training]['lab_str']),
                vec_diag.fit_transform(df.iloc[split.training]['diag_str'])
            ])
            y_train_sub = y[split.training]
            
            # 仅转换验证集
            X_calibration = hstack([
                vec_med.transform(df.iloc[split.calibration]['med_str']),
                vec_lab.transform(df.iloc[split.calibration]['lab_str']),
                vec_diag.transform(df.iloc[split.calibration]['diag_str'])
            ])
            y_calibration = y[split.calibration]
            
            # 仅转换测试集
            X_test = hstack([
                vec_med.transform(df.iloc[split.test]['med_str']),
                vec_lab.transform(df.iloc[split.test]['lab_str']),
                vec_diag.transform(df.iloc[split.test]['diag_str'])
            ])
            y_test = y[split.test]
            
            # 1. 训练基线模型
            model = clone(model_obj)
            model.fit(X_train_sub, y_train_sub)
            
            # 2. 在验证集上提取概率并寻找最佳温度 T
            calibration_probas = model.predict_proba(X_calibration)
            calibration_logits = probabilities_to_logits(calibration_probas)
            best_T = fit_temperature(calibration_logits, y_calibration)
            
            # 3. 在测试集上进行预测，并应用最优温度进行平滑校准
            test_probas_once = model.predict_proba(X_test)
            test_logits = probabilities_to_logits(test_probas_once)
            test_preds_raw = probabilities_from_logits(test_logits, best_T, "raw")
            test_preds_calib = probabilities_from_logits(
                test_logits, best_T, "calibrated"
            )
            
            # 4. 计算大满贯指标并更新字典
            metrics = calculate_sci_metrics_with_ci(
                y_test,
                test_preds_calib,
                n_bootstraps=1000,
                bootstrap_seed=derive_seed(
                    settings.reproducibility.bootstrap_seed, model_name, fold
                ),
            )
            metrics.update(
                {
                    "Model_Architecture": model_name,
                    "Fold": fold,
                    "Run_ID": args.run_id,
                    "Selected_Epoch": 0,
                    "Temperature": best_T,
                    "Probability_Mode": "calibrated",
                }
            )
            all_fold_results.append(metrics)
            
            print(f"   Fold {fold} | AUROC: {metrics['AUROC']:.4f} | Quantile ECE: {metrics['Quantile_ECE']:.4f}")
            
            # 5. 持久化校准后的测试集微观概率，用于后期画阴影图
            model_pred_dir = preds_dir / model_name
            model_pred_dir.mkdir(parents=True, exist_ok=True)
            pred_df = pd.DataFrame(
                {
                    "dataset_index": split.test,
                    "y_true": y_test,
                    "y_prob_raw": test_preds_raw,
                    "y_prob_calibrated": test_preds_calib,
                    "fold": fold,
                    "run_id": args.run_id,
                }
            )
            pred_df.to_csv(model_pred_dir / f"fold_{fold:02d}.csv", index=False)

            metadata = build_artifact_metadata(
                artifact_type="sklearn",
                run_id=args.run_id,
                model_name=model_name,
                fold=fold,
                selected_epoch=0,
                temperature=best_T,
                split_payload=split.checkpoint_payload(),
                dataset=data_fingerprint,
                configuration=model_config,
            )
            save_sklearn_artifact(
                sklearn_artifact_path(settings, args.run_id, model_name, fold),
                model,
                {"med": vec_med, "lab": vec_lab, "diagnosis": vec_diag},
                metadata,
            )

    report_path = metrics_dir / "ml_baselines.csv"
    df_results = pd.DataFrame(all_fold_results)
    df_results.to_csv(report_path, index=False)
    print(f"\nCalibrated ML baseline metrics saved to {report_path}")

if __name__ == "__main__":
    main()

"""
DILI-PLUS | 温度缩放后的传统机器学习基线（包实现）

职责：以 TF-IDF 表示训练 Logistic Regression 与 XGBoost；使用与深度模型相同的
training/selection/calibration/test 四方 grouped split，并仅对 outer test 推理一次。
输入：03_dili_dual_stream_tensors.parquet、03b_diag_tensors.parquet。
输出：run-specific 配对 raw/calibrated 预测、指标和版本化 sklearn artifact。
状态：当前 AHI-proxy 单任务的传统机器学习主评估路径。
说明：固定超参数基线不读取 selection partition；该分区仍保留以维持统一协议。
"""

import os
import argparse
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.base import clone
from scipy.sparse import hstack
import warnings
warnings.filterwarnings("ignore")

from diliplus.config import load_settings
from diliplus.data.lineage import validate_model_data_lineage
from diliplus.evaluation.metrics import compute_binary_metrics
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

def _formal_metrics(y_true, y_prob, reference_prevalence, settings):
    protocol = settings.evaluation_protocol
    dca_thresholds = np.arange(
        protocol.dca_min_threshold,
        protocol.dca_max_threshold + protocol.dca_step / 2.0,
        protocol.dca_step,
    )
    return compute_binary_metrics(
        y_true,
        y_prob,
        reference_prevalence=reference_prevalence,
        p_auc_fpr_limits=protocol.p_auc_fpr_limits,
        risk_thresholds=protocol.risk_thresholds,
        alert_budgets=protocol.alert_budgets,
        dca_thresholds=dca_thresholds,
    )

# =============================================================================
# 🌟 第三部分：数据加载与主控循环
# =============================================================================
def load_and_flatten_data(data_dir):
    print("[ML Baseline] Loading tensors and flattening to a TF-IDF corpus...")
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
    parser.add_argument("--run-kind", choices=("formal", "pilot_legacy"), default="formal")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--seed-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.run_kind == "pilot_legacy" and not args.run_id.startswith("pilot_legacy"):
        parser.error("pilot_legacy run IDs must start with 'pilot_legacy'")
    if args.run_kind == "formal" and args.max_folds is not None:
        parser.error("formal runs may not limit outer folds")
    if args.max_folds is not None and args.max_folds < 1:
        parser.error("max-folds must be positive")
    lineage = validate_model_data_lineage(settings, args.run_kind)
    data_dir = str(settings.model_data_dir)
    report_root = run_report_dir(settings, args.run_id)
    preds_dir = report_root / "predictions"
    metrics_dir = report_root / "metrics"
    preds_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_and_flatten_data(data_dir)
    y = df['label_ahi_proxy'].values
    
    models_to_train = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight=None,
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
    if "patient_id" not in df.columns or df["patient_id"].isna().any():
        raise KeyError("Formal ML data require non-null source patient_id groups")
    patient_ids = df["patient_id"].astype(str).to_numpy()
    labels = np.asarray(y, dtype=np.int64)
    folds = build_nested_grouped_splits(
        encounter_ids, labels, settings, group_ids=patient_ids
    )
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    data_fingerprint = dataset_fingerprint(settings)
    all_fold_results = []
    
    print("[ML Baseline] Starting 5-fold evaluation with temperature scaling...")
    for model_name, model_obj in models_to_train.items():
        print(f"\n{'='*50}\nEvaluating Model: {model_name}\n{'='*50}")
        
        model_config = config_snapshot(
            settings,
            {
                "model": model_name,
                "target_name": "label_ahi_proxy",
                "task_contract": "single_task_binary_classification",
                "representation": "TF-IDF",
                "estimator_parameters": model_obj.get_params(deep=False),
                "selection_partition_usage": "reserved_not_used_fixed_hyperparameters",
                "class_weighting": None,
                "run_kind": args.run_kind,
                "seed_index": args.seed_index,
                "data_lineage": lineage,
            },
        )
        for split in folds:
            fold = split.fold
            fold_started = time.perf_counter()
            
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
            reference_prevalence = float(y_train_sub.mean())
            fold_duration = time.perf_counter() - fold_started
            
            # 5. 持久化校准后的测试集微观概率，用于后期画阴影图
            model_pred_dir = preds_dir / model_name
            model_pred_dir.mkdir(parents=True, exist_ok=True)
            pred_df = pd.DataFrame(
                {
                    "dataset_index": split.test,
                    "encounter_id": encounter_ids[split.test],
                    "patient_id": patient_ids[split.test],
                    "y_true": y_test,
                    "logit_0": test_logits[:, 0],
                    "logit_1": test_logits[:, 1],
                    "y_prob_raw": test_preds_raw,
                    "y_prob_calibrated": test_preds_calib,
                    "fold": fold,
                    "run_id": args.run_id,
                    "run_kind": args.run_kind,
                    "seed_index": args.seed_index,
                    "reference_prevalence": reference_prevalence,
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

            for probability_mode, probabilities in (
                ("raw", test_preds_raw),
                ("calibrated", test_preds_calib),
            ):
                metrics = _formal_metrics(
                    y_test, probabilities, reference_prevalence, settings
                )
                metrics.update(
                    {
                        "Model_Architecture": model_name,
                        "Fold": fold,
                        "Run_ID": args.run_id,
                        "Run_Kind": args.run_kind,
                        "Seed_Index": args.seed_index,
                        "Selected_Epoch": 0,
                        "Temperature": best_T,
                        "Probability_Mode": probability_mode,
                        "Feature_Count": int(X_train_sub.shape[1]),
                        "Fold_Duration_Seconds": fold_duration,
                        "Device": "cpu",
                    }
                )
                all_fold_results.append(metrics)
            print(
                f"   Fold {fold} | calibrated AUROC: {all_fold_results[-1]['AUROC']:.4f} | "
                f"AUPRC: {all_fold_results[-1]['AUPRC']:.4f} | "
                f"duration: {fold_duration:.1f}s"
            )

    report_path = metrics_dir / "ml_baselines.csv"
    df_results = pd.DataFrame(all_fold_results)
    df_results.to_csv(report_path, index=False)
    print(f"\nCalibrated ML baseline metrics saved to {report_path}")

if __name__ == "__main__":
    main()

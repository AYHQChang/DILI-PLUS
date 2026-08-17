"""
DILI-PLUS | DILI 提前预警时间窗评估（包实现）

职责：在五折测试集上遮蔽序列终点前 0、24、48、72 小时的动态事件，比较四种
深度模型的 AUROC、AUPRC、校准与决策曲线指标随预警时间的变化。
输入：DILIPlusDataset 和指定 run ID 的版本化模型 artifact。
输出：reports/06a_Early_Warning_Decay_Results.csv。
状态：当前 DILI 单任务的时间窗敏感性评估脚本。
实现边界：调用方必须显式选择 raw 或 calibrated 概率；TextCNN 与逐 horizon 诊断/动态
截断仍需在 Code-09 修正后再解释。
"""

import os
import torch
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, roc_curve
from torch.utils.data import DataLoader, Subset

from diliplus.artifacts import (
    artifact_probabilities,
    dataset_fingerprint,
    deep_artifact_path,
    load_deep_artifact,
)
from diliplus.config import load_settings
from diliplus.data.dataset import DILIPlusDataset, load_vocab_sizes
from diliplus.models.diliplus_engine import DILIPlusEngine
from diliplus.models.baselines import MultiModalBiLSTM, MultiModalBaselineMedBERT, MultiModalTextCNN
from diliplus.reproducibility import DEFAULT_SEED, derive_seed, seed_everything

warnings.filterwarnings("ignore")

# =============================================================================
# 与训练结果表一致的评估指标
# =============================================================================
def calculate_partial_auc(y_true, y_prob, fpr_limit=0.2):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    if fpr_limit <= 0 or fpr_limit > 1: return 0.0
    valid_idx = np.where(fpr <= fpr_limit)[0]
    if len(valid_idx) < 2: return 0.0
    fpr_part, tpr_part = fpr[valid_idx], tpr[valid_idx]
    if fpr_part[-1] < fpr_limit:
        idx_next = valid_idx[-1] + 1
        if idx_next < len(fpr):
            slope = (tpr[idx_next] - tpr_part[-1]) / (fpr[idx_next] - fpr_part[-1] + 1e-9)
            tpr_interp = tpr_part[-1] + slope * (fpr_limit - fpr_part[-1])
            fpr_part = np.append(fpr_part, fpr_limit)
            tpr_part = np.append(tpr_part, tpr_interp)
    # 🔥 已经为您替换为 Numpy 2.0 强制要求的 trapezoid
    return np.trapezoid(tpr_part, fpr_part) / (fpr_limit * 1.0) 

def calculate_net_benefit(y_true, y_prob, thresholds=np.arange(0.01, 1.0, 0.01)):
    net_benefits = []
    n = len(y_true)
    if n == 0: return 0.0
    for pt in thresholds:
        preds = (y_prob >= pt).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits.append(nb)
    # 🔥 已经为您替换为 Numpy 2.0 强制要求的 trapezoid
    return np.trapezoid(np.maximum(net_benefits, 0), thresholds)

def calculate_quantile_ece(y_true, y_prob, n_bins=10):
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
    y_true, y_prob = np.array(y_true), np.array(y_prob)
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
# 🌟 核心引擎：物理时间截断器 (Temporal Masking Engine)
# =============================================================================
def apply_temporal_mask(mask, dt, lead_time_hours):
    """
    通过张量翻转和累加，计算每个 Token 距离终点的物理时间。
    将距离终点在 lead_time_hours 之内的事件强制致盲，实现真实的“提前预警”。
    """
    if lead_time_hours == 0:
        return mask
    
    # 将时间差反转，从最后一个事件向回累加计算距今时间
    dt_flipped = torch.flip(dt, dims=[1])
    cum_time_from_end = torch.cumsum(dt_flipped, dim=1)
    cum_time_from_end = torch.flip(cum_time_from_end, dims=[1])
    
    # 只允许“发生时间距离终点 > 预警窗口”的事件可见
    valid_time_mask = cum_time_from_end >= lead_time_hours
    return mask & valid_time_mask

# =============================================================================
# 🌟 模型预警评估管线
# =============================================================================
@torch.no_grad()
def evaluate_lead_time_single_fold(
    model,
    dataloader,
    device,
    lead_time_hours,
    artifact_metadata,
    probability_mode,
    bootstrap_seed=DEFAULT_SEED,
):
    model.eval()
    all_preds, all_labels = [], []
    
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if 'label' not in k}
        labels = batch.get('label_dili', batch.get('label')).to(device)
        
        # 根据预警窗口更新动态序列掩码
        if lead_time_hours > 0:
            # 1. 截断注意力掩码
            valid_mask_med = apply_temporal_mask(inputs['mask_med'], inputs['dt_med'], lead_time_hours)
            valid_mask_lab = apply_temporal_mask(inputs['mask_lab'], inputs['dt_lab'], lead_time_hours)
            
            inputs['mask_med'] = valid_mask_med
            inputs['mask_lab'] = valid_mask_lab
            
            # 2. 旧版输入键兼容分支；现行数据集使用 x_med/x_lab/v_lab 键，
            # 因而本段不会为 TextCNN 清零实际特征，后续逻辑修正时需要对齐键名。
            # 将被截断的离散 Token 替换为 0 (通常 0 是 PAD token)
            if 'med' in inputs:
                inputs['med'] = inputs['med'] * valid_mask_med.long()
            if 'lab' in inputs:
                inputs['lab'] = inputs['lab'] * valid_mask_lab.long()
                
            # 对旧版 lab_val 键同步应用掩码
            if 'lab_val' in inputs:
                # 扩展掩码以匹配最后一个维度 (如 lab_val 为 3D)
                extended_mask_lab = valid_mask_lab.unsqueeze(-1) if inputs['lab_val'].dim() > valid_mask_lab.dim() else valid_mask_lab
                inputs['lab_val'] = inputs['lab_val'] * extended_mask_lab.float()
            
        outputs = model(**inputs)
        
        # 兼容 tuple、dict 与直接 logits 三种模型输出
        if isinstance(outputs, tuple) and len(outputs) == 2: logits = outputs[1]
        elif isinstance(outputs, dict) and "logits" in outputs: logits = outputs["logits"]
        else: logits = outputs
            
        probs = artifact_probabilities(
            logits.detach().cpu(), artifact_metadata, probability_mode
        )
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        
    return calculate_sci_metrics_with_ci(
        all_labels,
        all_preds,
        n_bootstraps=500,
        bootstrap_seed=bootstrap_seed,
    )

def main(settings=None, run_id=None, probability_mode="calibrated"):
    settings = settings or load_settings()
    if not run_id:
        raise ValueError("run_id is required to resolve versioned model artifacts")
    if probability_mode not in ("raw", "calibrated"):
        raise ValueError("probability_mode must be 'raw' or 'calibrated'")
    seed_everything(settings.reproducibility)
    data_dir = str(settings.model_data_dir)
    vocab_dir = str(settings.paths.vocab)
    report_dir = str(settings.paths.reports)
    os.makedirs(report_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Booting DILIPLUS Early Warning Simulator | Device: {device}")
    
    vocab_config = load_vocab_sizes(vocab_dir)
    full_dataset = DILIPlusDataset(data_dir, vocab_dir)
    
    current_fingerprint = dataset_fingerprint(settings)["payload_sha256"]

    models_to_evaluate = {
        "MultiModalTextCNN": MultiModalTextCNN,
        "MultiModalBiLSTM": MultiModalBiLSTM,
        "MultiModalBaselineMedBERT": MultiModalBaselineMedBERT,
        "MultiModalTimeAwareMedBERT": DILIPlusEngine
    }
    
    lookahead_windows = [0, 24, 48, 72] # 预警窗口：即刻、提前1天、2天、3天
    all_results = []
    
    for model_name, model_class in models_to_evaluate.items():
        print(f"\n{'='*60}\n⏳ Evaluating Decay for: {model_name}\n{'='*60}")
        
        for lead_time in lookahead_windows:
            fold_metrics = []
            
            # 执行 5 折全局评估
            for fold in range(1, settings.evaluation_protocol.outer_folds + 1):
                artifact_path = deep_artifact_path(settings, run_id, model_name, fold)
                if not artifact_path.exists():
                    print(f"   Fold {fold} artifact not found, skipping...")
                    continue
                
                model = model_class(**vocab_config).to(device)
                metadata = load_deep_artifact(
                    artifact_path,
                    model,
                    map_location=device,
                    expected_run_id=run_id,
                    expected_model_name=model_name,
                    expected_fold=fold,
                    expected_dataset_fingerprint=current_fingerprint,
                )
                test_idx = metadata["split"]["indices"]["test"]
                test_loader = DataLoader(
                    Subset(full_dataset, test_idx),
                    batch_size=settings.training.batch_size,
                    shuffle=False,
                    num_workers=settings.reproducibility.dataloader_num_workers,
                )
                
                metrics = evaluate_lead_time_single_fold(
                    model,
                    test_loader,
                    device,
                    lead_time,
                    metadata,
                    probability_mode,
                    bootstrap_seed=derive_seed(
                        settings.reproducibility.bootstrap_seed,
                        model_name,
                        fold,
                        lead_time,
                    ),
                )
                fold_metrics.append(metrics)
            
            if len(fold_metrics) == 0:
                continue
                
            # 汇总 5 折均值
            avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()}
            avg_metrics['Model_Architecture'] = model_name
            avg_metrics['Lead_Time_Hours'] = lead_time
            avg_metrics['Run_ID'] = run_id
            avg_metrics['Probability_Mode'] = probability_mode
            all_results.append(avg_metrics)
            
            print(f"   🕒 {lead_time:>2}h Ahead | Avg AUROC: {avg_metrics['AUROC']:.4f} | Avg AUPRC: {avg_metrics['AUPRC']:.4f}")

    # 保存预警衰减结果矩阵
    df_results = pd.DataFrame(all_results)
    
    # 调整列序美观
    cols = ['Run_ID', 'Probability_Mode', 'Model_Architecture', 'Lead_Time_Hours', 'AUROC', 'AUROC_95CI_Lower', 'AUROC_95CI_Upper',
            'AUPRC', 'Quantile_ECE', 'pAUC_0.2', 'NetBenefit_AUDC', 'Brier']
    df_results = df_results[[c for c in cols if c in df_results.columns]]
    
    out_path = os.path.join(
        report_dir,
        "runs",
        run_id,
        f"06a_Early_Warning_Decay_Results_{probability_mode}.csv",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"\n✅ All Decay Simulations Completed! Data saved to: {out_path}")

if __name__ == "__main__":
    main()

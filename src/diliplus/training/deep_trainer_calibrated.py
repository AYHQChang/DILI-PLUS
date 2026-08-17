"""
DILI-PLUS | 深度学习训练与后置温度缩放（包实现）

职责：对四种深度模型执行四方 grouped protocol；training 只拟合参数，selection
只用于早停，calibration 只拟合温度，outer test 只执行一次最终 logits 推理。
输入：DILIPlusDataset、词表和模型定义。
输出：checkpoints/runs/<run_id>/ 下的版本化 artifact，以及同 run 的配对 raw/calibrated
测试概率与指标。
状态：当前 AHI-proxy 单任务的深度学习主训练路径。
实现边界：所有正式模型从零初始化并返回单一 ``[batch, 2]`` logits；训练使用
无类别权重的 focal modulation，不包含 AKI、多任务不确定性加权或自校准。
"""

import os
import torch
import torch.optim as optim
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from torch.utils.data import DataLoader, Subset

from diliplus.artifacts import (
    build_artifact_metadata,
    config_snapshot,
    dataset_fingerprint,
    deep_artifact_path,
    run_report_dir,
    save_deep_artifact,
)
from diliplus.calibration import (
    fit_temperature,
    probabilities_from_logits,
)
from diliplus.config import load_settings
from diliplus.data.dataset import DILIPlusDataset, load_vocab_sizes
from diliplus.models.registry import (
    FORMAL_DEEP_MODEL_NAMES,
    build_formal_deep_model,
    extract_ahi_proxy_logits,
)
from diliplus.reproducibility import (
    DEFAULT_SEED,
    derive_seed,
    make_torch_generator,
    seed_dataloader_worker,
    seed_everything,
)
from diliplus.splits import build_nested_grouped_splits
from diliplus.training.losses import UnweightedFocalLoss

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 第二部分：现有评估指标计算（正式统计合同将在 Code-10 完成）
# =============================================================================
def calculate_partial_auc(y_true, y_prob, fpr_limit=0.2):
    """计算 FPR <= 0.2 区间内的 Partial AUC，更符合高特异性临床需求。"""
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
            
    pauc = np.trapezoid(tpr_part, fpr_part)
    max_pauc = fpr_limit * 1.0 
    return pauc / max_pauc

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
    return np.trapezoid(np.maximum(net_benefits, 0), thresholds)

def calculate_quantile_ece(y_true, y_prob, n_bins=10):
    """等频分箱的校准误差 (Quantile ECE)，对极端不平衡数据更稳健"""
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
    """汇总所有指标并计算 Bootstrap 95% 置信区间"""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = calculate_quantile_ece(y_true, y_prob)
    pauc = calculate_partial_auc(y_true, y_prob)
    audc = calculate_net_benefit(y_true, y_prob)
    
    bootstrapped_auroc = []
    rng = np.random.default_rng(bootstrap_seed)
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
# 第三部分：单任务深度学习训练与推理逻辑
# =============================================================================
def _ahi_proxy_labels(batch, device):
    if "label_ahi_proxy" in batch:
        return batch["label_ahi_proxy"].to(device)
    if "label" in batch:
        return batch["label"].to(device)
    raise KeyError("Batch is missing the required label_ahi_proxy tensor")


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        # 分离特征与标签
        inputs = {k: v.to(device) for k, v in batch.items() if 'label' not in k}
        
        labels = _ahi_proxy_labels(batch, device)
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        
        logits = extract_ahi_proxy_logits(outputs)
        loss = criterion(logits, labels)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

@torch.no_grad()
def collect_logits(model, dataloader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if 'label' not in k}
        labels = _ahi_proxy_labels(batch, device)
        
        outputs = model(**inputs)
        
        logits = extract_ahi_proxy_logits(outputs)
            
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return np.concatenate(all_logits, axis=0), np.asarray(all_labels, dtype=np.int64)


def evaluate(
    model, dataloader, device, return_raw=False, bootstrap_seed=DEFAULT_SEED
):
    logits, all_labels = collect_logits(model, dataloader, device)
    all_preds = probabilities_from_logits(logits, 1.0, "raw")
    if return_raw:
        return np.asarray(all_preds), np.asarray(all_labels)
    metrics = calculate_sci_metrics_with_ci(
        all_labels,
        all_preds,
        n_bootstraps=100,
        bootstrap_seed=bootstrap_seed,
    )
    return metrics, all_preds.tolist(), all_labels.tolist()

# =============================================================================
# 第四部分：四方 grouped 主控管线
# =============================================================================
def main(argv=None, settings=None):
    settings = settings or load_settings()
    seed_everything(settings.reproducibility)
    parser = argparse.ArgumentParser(
        description="DILI-PLUS single-task AHI-proxy trainer"
    )
    parser.add_argument(
        '--model', choices=FORMAL_DEEP_MODEL_NAMES, required=True
    )
    parser.add_argument('--epochs', type=int, default=settings.training.epochs)
    parser.add_argument('--batch_size', type=int, default=settings.training.batch_size)
    parser.add_argument('--lr', type=float, default=settings.training.learning_rate)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args(argv)

    CONFIG = {
        "data_dir": str(settings.model_data_dir),
        "vocab_dir": str(settings.paths.vocab),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": settings.training.weight_decay,
        "focal_gamma": settings.training.focal_gamma,
        "global_seed": settings.reproducibility.global_seed,
        "split_seed": settings.reproducibility.split_seed,
        "bootstrap_seed": settings.reproducibility.bootstrap_seed,
        "dataloader_num_workers": settings.reproducibility.dataloader_num_workers,
    }
    report_root = run_report_dir(settings, args.run_id)
    preds_dir = report_root / "predictions" / args.model
    metrics_dir = report_root / "metrics"
    preds_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_config = load_vocab_sizes(CONFIG["vocab_dir"])
    full_dataset = DILIPlusDataset(CONFIG["data_dir"], CONFIG["vocab_dir"])
    
    encounter_ids = full_dataset.data["encounter_id"].astype(str).to_numpy()
    labels = np.asarray(full_dataset.labels, dtype=np.int64)
    folds = build_nested_grouped_splits(encounter_ids, labels, settings)
    data_fingerprint = dataset_fingerprint(settings)
    training_config = config_snapshot(
        settings,
        {
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": settings.training.weight_decay,
            "loss_name": "UnweightedFocalLoss",
            "focal_gamma": settings.training.focal_gamma,
            "class_weighting": None,
            "target_name": "label_ahi_proxy",
            "task_contract": "single_task_binary_classification",
            "output_contract": "dict_with_logits_batch_by_2",
            "initialization": "from_scratch_no_external_pretraining",
            "external_pretraining": None,
            "hidden_size": settings.training.hidden_size,
            "num_heads": settings.training.num_heads,
            "dropout": settings.training.dropout,
            "diagnosis_modality_dropout_prob": (
                settings.training.diagnosis_modality_dropout_prob
            ),
        },
    )
    
    all_fold_results = []
    print(f"\n==================================================")
    print(f"[DILI-PLUS] Training single-task model: {args.model} | Device: {device}")
    print(f"==================================================")

    for split in folds:
        fold = split.fold
        print(f"\n--- Fold {fold}/{len(folds)} ---")
        fold_seed = derive_seed(CONFIG["global_seed"], args.model, fold)
        seed_everything(fold_seed, settings.reproducibility.deterministic_torch)

        train_loader = DataLoader(
            Subset(full_dataset, split.training),
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            generator=make_torch_generator(fold_seed),
            worker_init_fn=seed_dataloader_worker,
            num_workers=CONFIG["dataloader_num_workers"],
        )
        selection_loader = DataLoader(
            Subset(full_dataset, split.selection),
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["dataloader_num_workers"],
        )
        calibration_loader = DataLoader(
            Subset(full_dataset, split.calibration),
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["dataloader_num_workers"],
        )
        test_loader = DataLoader(
            Subset(full_dataset, split.test),
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["dataloader_num_workers"],
        )

        model = build_formal_deep_model(
            args.model, vocab_config, settings.training
        ).to(device)
        
        criterion = UnweightedFocalLoss(
            gamma=CONFIG["focal_gamma"]
        ).to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        best_auroc = float("-inf")
        best_state = None
        selected_epoch = 0
        patience_limit = 7
        patience_counter = 0

        # --- 阶段 1：深度学习主干训练 ---
        for epoch in range(CONFIG["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            metrics, _, _ = evaluate(
                model,
                selection_loader,
                device,
                bootstrap_seed=derive_seed(
                    CONFIG["bootstrap_seed"], args.model, fold, "selection"
                ),
            )
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   Ep [{epoch+1}/{CONFIG['epochs']}] LR: {current_lr:.6f} | Loss: {train_loss:.4f} | Val AUC: {metrics['AUROC']:.4f}")
            
            scheduler.step(metrics['AUROC'])
            
            if metrics['AUROC'] > best_auroc:
                best_auroc = metrics['AUROC']
                selected_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience_limit:
                print(f"   Early stopping: no improvement for {patience_limit} epochs.")
                break
        
        if best_state is None:
            raise RuntimeError(f"Fold {fold} did not produce a selected checkpoint")
        model.load_state_dict(best_state, strict=True)
        model.to(device)

        # Calibration is a separate role and is never used for epoch selection.
        calibration_logits, calibration_labels = collect_logits(
            model, calibration_loader, device
        )
        best_T = fit_temperature(calibration_logits, calibration_labels)
        print(f"      Fold {fold} calibration temperature = {best_T:.4f}")

        # The outer test loader is consumed exactly once. Raw and calibrated
        # probabilities are two views of this same in-memory logits array.
        test_logits, test_labels = collect_logits(model, test_loader, device)
        test_preds_raw = probabilities_from_logits(test_logits, best_T, "raw")
        test_preds_calib = probabilities_from_logits(test_logits, best_T, "calibrated")
        
        final_metrics = calculate_sci_metrics_with_ci(
            test_labels,
            test_preds_calib,
            n_bootstraps=1000,
            bootstrap_seed=derive_seed(
                CONFIG["bootstrap_seed"], args.model, fold, "final"
            ),
        )
        print(f"Calibrated Fold {fold} | AUROC: {final_metrics['AUROC']:.4f} | Quantile ECE: {final_metrics['Quantile_ECE']:.4f}")

        pred_df = pd.DataFrame(
            {
                "dataset_index": split.test,
                "y_true": test_labels,
                "y_prob_raw": test_preds_raw,
                "y_prob_calibrated": test_preds_calib,
                "fold": fold,
                "run_id": args.run_id,
            }
        )
        pred_df.to_csv(preds_dir / f"fold_{fold:02d}.csv", index=False)

        metadata = build_artifact_metadata(
            artifact_type="torch",
            run_id=args.run_id,
            model_name=args.model,
            fold=fold,
            selected_epoch=selected_epoch,
            temperature=best_T,
            split_payload=split.checkpoint_payload(),
            dataset=data_fingerprint,
            configuration=training_config,
        )
        save_deep_artifact(
            deep_artifact_path(settings, args.run_id, args.model, fold),
            model,
            metadata,
        )

        final_metrics.update(
            {
                "Model_Architecture": args.model,
                "Fold": fold,
                "Run_ID": args.run_id,
                "Selected_Epoch": selected_epoch,
                "Temperature": best_T,
                "Probability_Mode": "calibrated",
            }
        )
        all_fold_results.append(final_metrics)

    # One run-specific file per model; never append rows from older runs.
    report_path = metrics_dir / f"{args.model}.csv"
    df_results = pd.DataFrame(all_fold_results)
    df_results.to_csv(report_path, index=False)
    print(f"Calibration complete. Metrics saved to {report_path}")

if __name__ == "__main__":
    main()

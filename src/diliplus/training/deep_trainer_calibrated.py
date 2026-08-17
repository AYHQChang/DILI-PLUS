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

import time
import torch
import torch.optim as optim
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
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
from diliplus.data.lineage import validate_model_data_lineage
from diliplus.evaluation.metrics import compute_binary_metrics
from diliplus.models.registry import (
    DEEP_EXPERIMENT_NAMES,
    build_deep_experiment_model,
    experiment_spec,
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

def _dca_thresholds(settings):
    protocol = settings.evaluation_protocol
    return np.arange(
        protocol.dca_min_threshold,
        protocol.dca_max_threshold + protocol.dca_step / 2.0,
        protocol.dca_step,
    )


def _formal_metrics(y_true, y_prob, reference_prevalence, settings):
    protocol = settings.evaluation_protocol
    return compute_binary_metrics(
        y_true,
        y_prob,
        reference_prevalence=reference_prevalence,
        p_auc_fpr_limits=protocol.p_auc_fpr_limits,
        risk_thresholds=protocol.risk_thresholds,
        alert_budgets=protocol.alert_budgets,
        dca_thresholds=_dca_thresholds(settings),
    )

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


def evaluate(model, dataloader, device, return_raw=False, bootstrap_seed=DEFAULT_SEED):
    logits, all_labels = collect_logits(model, dataloader, device)
    all_preds = probabilities_from_logits(logits, 1.0, "raw")
    if return_raw:
        return np.asarray(all_preds), np.asarray(all_labels)
    metrics = {
        "AUROC": float(roc_auc_score(all_labels, all_preds)),
        "AUPRC": float(average_precision_score(all_labels, all_preds)),
    }
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
        '--model', choices=DEEP_EXPERIMENT_NAMES, required=True
    )
    parser.add_argument('--epochs', type=int, default=settings.training.epochs)
    parser.add_argument('--batch_size', type=int, default=settings.training.batch_size)
    parser.add_argument('--lr', type=float, default=settings.training.learning_rate)
    parser.add_argument('--run-id', required=True)
    parser.add_argument(
        '--run-kind', choices=('formal', 'pilot_legacy'), default='formal'
    )
    parser.add_argument(
        '--max-folds', type=int, default=None,
        help='Pilot-only limit; formal runs must evaluate all configured outer folds.',
    )
    parser.add_argument('--seed-index', type=int, default=0)
    args = parser.parse_args(argv)
    if args.run_kind == 'pilot_legacy' and not args.run_id.startswith('pilot_legacy'):
        parser.error("pilot_legacy run IDs must start with 'pilot_legacy'")
    if args.run_kind == 'formal' and args.max_folds is not None:
        parser.error("formal runs may not limit outer folds")
    if args.max_folds is not None and args.max_folds < 1:
        parser.error("max-folds must be positive")
    lineage = validate_model_data_lineage(settings, args.run_kind)

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
        "run_kind": args.run_kind,
        "seed_index": args.seed_index,
    }
    report_root = run_report_dir(settings, args.run_id)
    preds_dir = report_root / "predictions" / args.model
    metrics_dir = report_root / "metrics"
    history_dir = report_root / "history" / args.model
    preds_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_config = load_vocab_sizes(CONFIG["vocab_dir"])
    full_dataset = DILIPlusDataset(CONFIG["data_dir"], CONFIG["vocab_dir"])
    
    encounter_ids = full_dataset.data["encounter_id"].astype(str).to_numpy()
    if (
        "patient_id" not in full_dataset.data.columns
        or full_dataset.data["patient_id"].isna().any()
    ):
        raise KeyError("Formal deep-model data require non-null source patient_id groups")
    patient_ids = full_dataset.data["patient_id"].astype(str).to_numpy()
    labels = np.asarray(full_dataset.labels, dtype=np.int64)
    folds = build_nested_grouped_splits(
        encounter_ids, labels, settings, group_ids=patient_ids
    )
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
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
            "selection_metric": settings.training.selection_metric,
            "early_stopping_patience": settings.training.early_stopping_patience,
            "scheduler_patience": settings.training.scheduler_patience,
            "run_kind": args.run_kind,
            "seed_index": args.seed_index,
            "data_lineage": lineage,
            "model_input_spec": experiment_spec(args.model),
        },
    )
    
    all_fold_results = []
    print(f"\n==================================================")
    print(f"[DILI-PLUS] Training single-task model: {args.model} | Device: {device}")
    print(f"==================================================")

    for split in folds:
        fold = split.fold
        print(f"\n--- Fold {fold}/{len(folds)} ---")
        fold_seed = derive_seed(
            CONFIG["global_seed"], args.model, fold, "seed_index", args.seed_index
        )
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

        model = build_deep_experiment_model(
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
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=settings.training.scheduler_patience,
        )

        best_auprc = float("-inf")
        best_state = None
        selected_epoch = 0
        patience_limit = settings.training.early_stopping_patience
        patience_counter = 0
        fold_started = time.perf_counter()
        history_rows = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

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
            history_rows.append(
                {
                    "epoch": epoch + 1,
                    "learning_rate": current_lr,
                    "training_loss": train_loss,
                    "selection_AUROC": metrics['AUROC'],
                    "selection_AUPRC": metrics['AUPRC'],
                }
            )
            print(
                f"   Ep [{epoch+1}/{CONFIG['epochs']}] LR: {current_lr:.6f} | "
                f"Loss: {train_loss:.4f} | Selection AUPRC: {metrics['AUPRC']:.4f} | "
                f"AUROC: {metrics['AUROC']:.4f}"
            )
            
            scheduler.step(metrics['AUPRC'])
            
            if metrics['AUPRC'] > best_auprc:
                best_auprc = metrics['AUPRC']
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
        pd.DataFrame(history_rows).to_csv(
            history_dir / f"fold_{fold:02d}.csv", index=False
        )

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
        
        reference_prevalence = float(labels[split.training].mean())
        fold_duration = time.perf_counter() - fold_started
        peak_gpu_mb = (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        )

        pred_df = pd.DataFrame(
            {
                "dataset_index": split.test,
                "encounter_id": encounter_ids[split.test],
                "patient_id": patient_ids[split.test],
                "y_true": test_labels,
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

        for probability_mode, probabilities in (
            ("raw", test_preds_raw),
            ("calibrated", test_preds_calib),
        ):
            final_metrics = _formal_metrics(
                test_labels, probabilities, reference_prevalence, settings
            )
            final_metrics.update(
                {
                    "Model_Architecture": args.model,
                    "Fold": fold,
                    "Run_ID": args.run_id,
                    "Run_Kind": args.run_kind,
                    "Seed_Index": args.seed_index,
                    "Selected_Epoch": selected_epoch,
                    "Best_Selection_AUPRC": best_auprc,
                    "Temperature": best_T,
                    "Probability_Mode": probability_mode,
                    "Parameter_Count": int(
                        sum(parameter.numel() for parameter in model.parameters())
                    ),
                    "Fold_Duration_Seconds": fold_duration,
                    "Peak_GPU_Memory_MB": peak_gpu_mb,
                    "Device": str(device),
                }
            )
            all_fold_results.append(final_metrics)
        print(
            f"Fold {fold} | calibrated AUROC: {all_fold_results[-1]['AUROC']:.4f} | "
            f"AUPRC: {all_fold_results[-1]['AUPRC']:.4f} | "
            f"duration: {fold_duration:.1f}s | peak GPU: {peak_gpu_mb:.1f} MB"
        )

    # One run-specific file per model; never append rows from older runs.
    report_path = metrics_dir / f"{args.model}.csv"
    df_results = pd.DataFrame(all_fold_results)
    df_results.to_csv(report_path, index=False)
    print(f"Calibration complete. Metrics saved to {report_path}")

if __name__ == "__main__":
    main()

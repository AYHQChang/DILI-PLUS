"""
DILI-PLUS | 未校准深度学习对照训练器（包实现）

职责：以统一九输入接口训练 TextCNN、BiLSTM、Baseline MedBERT 和时间感知
MedBERT，执行五折 GroupKFold 并保存每折预测与权重。
输入：DILIPlusDataset、词表和模型定义。
输出：未校准 checkpoints、reports/predictions/ 和 05_Experiment_Results_Table.csv。
状态：历史未校准对照路径。当前实现按测试折 AUROC 选择 epoch，因此不应作为
最终无偏性能报告的训练入口；主实验优先使用 calibrated 版本。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset

from diliplus.config import load_settings
from diliplus.data.dataset import DILIPlusDataset, load_vocab_sizes
from diliplus.models.diliplus_engine import DILIPlusEngine
from diliplus.models.baselines import MultiModalTextCNN, MultiModalBiLSTM, MultiModalBaselineMedBERT

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 🌟 高阶临床指标与置信区间计算 (SCI 级配置)
# ---------------------------------------------------------
def calculate_ece(y_true, y_prob, n_bins=10):
    """计算期望校准误差 (Expected Calibration Error)"""
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
    """计算核心指标并使用 Bootstrap 获取 95% 置信区间"""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = calculate_ece(y_true, y_prob)
    
    # Bootstrapping for CI
    rng_seed = 42
    bootstrapped_auroc = []
    rng = np.random.RandomState(rng_seed)
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2:
            continue
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
# 🌟 训练与评估逻辑
# ---------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        logits = outputs["logits"] 
        
        loss = criterion(logits, labels)
        loss.backward()
        
        # 🔥 FIX 4: 梯度裁剪 (Gradient Clipping)，防止任何异常张量毒化模型权重
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        labels = batch['label'].to(device)
        
        outputs = model(**inputs)
        probs = torch.softmax(outputs["logits"], dim=1)[:, 1]
        
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    metrics = calculate_sci_metrics_with_ci(all_labels, all_preds, n_bootstraps=100) # 训练期调小提速
    return metrics, all_preds, all_labels # 🔥 增加返回真实标签和预测概率

# ---------------------------------------------------------
# 🌟 主控函数
# ---------------------------------------------------------
def main(argv=None, settings=None):
    settings = settings or load_settings()
    parser = argparse.ArgumentParser(description="DILIPLUS Trainer (V13 SCI Full)")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=settings.training.epochs)
    parser.add_argument('--batch_size', type=int, default=settings.training.batch_size)
    parser.add_argument('--lr', type=float, default=settings.training.learning_rate)
    args = parser.parse_args(argv)

    CONFIG = {
        "data_dir": str(settings.paths.data_cache),
        "vocab_dir": str(settings.paths.vocab),
        "save_dir": str(settings.paths.checkpoints),
        "report_dir": str(settings.paths.reports),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr
    }
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    os.makedirs(CONFIG["report_dir"], exist_ok=True)
    
    # 🔥 建立预测概率存储专区
    preds_dir = os.path.join(CONFIG["report_dir"], "predictions")
    os.makedirs(preds_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_registry = {
        "MultiModalTextCNN": MultiModalTextCNN,
        "MultiModalBiLSTM": MultiModalBiLSTM,
        "MultiModalBaselineMedBERT": MultiModalBaselineMedBERT,
        "MultiModalTimeAwareMedBERT": DILIPlusEngine,
        "DILIPlus": DILIPlusEngine
    }
    
    vocab_config = load_vocab_sizes(CONFIG["vocab_dir"])
    full_dataset = DILIPlusDataset(CONFIG["data_dir"], CONFIG["vocab_dir"])
    
    groups = [str(full_dataset.data.iloc[i]['encounter_id']).split('_')[0] for i in range(len(full_dataset))]
    gkf = GroupKFold(n_splits=5)
    
    all_fold_results = []
    print(f"\n==================================================")
    print(f"🚀 Booting Model: {args.model} | Device: {device}")
    print(f"==================================================")

    for fold, (train_idx, test_idx) in enumerate(gkf.split(full_dataset, groups=groups)):
        print(f"\n--- Fold {fold+1}/5 ---")
        train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=CONFIG["batch_size"], shuffle=True)
        test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=CONFIG["batch_size"], shuffle=False)

        model = model_registry[args.model](
            vocab_med_size=vocab_config["vocab_med_size"],
            vocab_lab_size=vocab_config["vocab_med_size"], 
            vocab_diag_size=vocab_config["vocab_diag_size"]
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-5)
        
        # 修复PyTorch 2.2+版本问题，无 verbose 参数
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        best_auroc = 0.0
        best_metrics = {}
        best_preds_df = None # 🔥 暂存每折最佳预测结果
        
        patience_limit = 7
        patience_counter = 0

        for epoch in range(CONFIG["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            metrics, preds, labels = evaluate(model, test_loader, device) # 🔥 接收预测结果
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   Ep [{epoch+1}/{CONFIG['epochs']}] LR: {current_lr:.6f} | Loss: {train_loss:.4f} | AUC: {metrics['AUROC']:.4f} | PRC: {metrics['AUPRC']:.4f}")
            
            scheduler.step(metrics['AUROC'])
            
            if metrics['AUROC'] > best_auroc:
                best_auroc = metrics['AUROC']
                best_metrics = metrics.copy()
                
                # 🔥 捕获创新高时的概率分布
                best_preds_df = pd.DataFrame({'y_true': labels, 'y_prob': preds, 'fold': fold+1})
                
                torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], f"best_{args.model}_Fold{fold+1}.pth"))
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience_limit:
                print(f"   🛑 Early Stopping triggered! No improvement for {patience_limit} epochs.")
                break
        
        print(f"✅ Fold {fold+1} Best AUROC: {best_metrics['AUROC']:.4f} (95% CI: {best_metrics['AUROC_95CI_Lower']:.4f}-{best_metrics['AUROC_95CI_Upper']:.4f}) | Brier: {best_metrics['Brier']:.4f}")
        
        # 🔥 单折训练完毕，将最佳概率落地到本地 CSV
        if best_preds_df is not None:
            best_preds_df.to_csv(os.path.join(preds_dir, f"preds_{args.model}_Fold{fold+1}.csv"), index=False)
            
        best_metrics.update({"Model_Architecture": args.model, "Fold": fold+1})
        all_fold_results.append(best_metrics)

    # 汇总写入主表
    report_path = os.path.join(CONFIG["report_dir"], "05_Experiment_Results_Table.csv")
    df_results = pd.DataFrame(all_fold_results)
    if os.path.exists(report_path):
        df_existing = pd.read_csv(report_path)
        df_results = pd.concat([df_existing, df_results], ignore_index=True)
    df_results.to_csv(report_path, index=False)
    print(f"📊 {args.model} Complete. Metrics saved to {report_path}")

if __name__ == "__main__":
    main()

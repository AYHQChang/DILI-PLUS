"""
DILI-PLUS | 深度学习训练与后置温度缩放（包实现）

职责：对四种深度模型执行五折 GroupKFold；每个外层训练折再划分训练/验证集，
验证集同时用于早停与温度参数拟合，外层测试折用于最终评价。
输入：DILIPlusDataset、词表和模型定义。
输出：best_calib_*.pth、reports/predictions_calibrated/ 和校准结果表。
状态：当前 DILI 单任务的深度学习主训练路径。
实现边界：当前模型返回单一 DILI 输出头，实际训练使用 DILI FocalLoss；文件中保留的
AKI/多任务兼容分支不会被现行 DILIPlusEngine 激活。温度参数未写入 .pth 权重文件。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from diliplus.config import load_settings
from diliplus.data.dataset import DILIPlusDataset, load_vocab_sizes
from diliplus.models.diliplus_engine import DILIPlusEngine
from diliplus.models.baselines import MultiModalTextCNN, MultiModalBiLSTM, MultiModalBaselineMedBERT

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 第一部分：DILI Focal Loss 与保留的多任务兼容损失
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class UncertaintyMTLLoss(nn.Module):
    """
    通过同方差不确定性(Homoscedastic Uncertainty)自适应调节多任务的损失权重。
    """
    def __init__(self, num_tasks=2):
        super(UncertaintyMTLLoss, self).__init__()
        # 初始化为0，即可学习的 log(sigma^2)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
        # 针对不同发病率设定特定的 FocalLoss
        self.loss_fn_aki = FocalLoss(alpha=0.3, gamma=2.0)
        self.loss_fn_dili = FocalLoss(alpha=0.25, gamma=2.0)

    def forward(self, logits_aki, logits_dili, targets_aki, targets_dili):
        loss_0 = self.loss_fn_aki(logits_aki, targets_aki)
        loss_1 = self.loss_fn_dili(logits_dili, targets_dili)
        
        precision_0 = torch.exp(-self.log_vars[0])
        precision_1 = torch.exp(-self.log_vars[1])
        
        loss = precision_0 * loss_0 + self.log_vars[0] + \
               precision_1 * loss_1 + self.log_vars[1]
        
        return loss, loss_0.item(), loss_1.item()


# =============================================================================
# 🌟 修复增补：跨界温度缩放器 (DL Logits -> L-BFGS -> Scaled Probabilities)
# =============================================================================
class TemperatureScaler(nn.Module):
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

def fit_temperature_scaling(val_probas, val_labels):
    eps = 1e-9
    val_probas = np.clip(val_probas, eps, 1.0 - eps)
    val_logits = np.stack([np.log(1 - val_probas), np.log(val_probas)], axis=1)
    
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
    return scaler.temperature.item()

def apply_temperature_scaling(test_probas, T):
    eps = 1e-9
    test_probas = np.clip(test_probas, eps, 1.0 - eps)
    test_logits = np.stack([np.log(1 - test_probas), np.log(test_probas)], axis=1)
    
    scaled_logits = torch.tensor(test_logits, dtype=torch.float32) / T
    calibrated_probas = torch.softmax(scaled_logits, dim=1).numpy()
    return calibrated_probas[:, 1]


# =============================================================================
# 🌟 第二部分：顶刊级高阶评估指标计算
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

def calculate_sci_metrics_with_ci(y_true, y_prob, n_bootstraps=1000):
    """汇总所有指标并计算 Bootstrap 95% 置信区间"""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = calculate_quantile_ece(y_true, y_prob)
    pauc = calculate_partial_auc(y_true, y_prob)
    audc = calculate_net_benefit(y_true, y_prob)
    
    rng_seed = 42
    bootstrapped_auroc = []
    rng = np.random.RandomState(rng_seed)
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
# 🌟 第三部分：深度学习训练与推理逻辑
# =============================================================================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        # 分离特征与标签
        inputs = {k: v.to(device) for k, v in batch.items() if 'label' not in k}
        
        # 兼容性处理：无论数据集吐出的是 label_dili 还是 label，都能无缝接管
        labels_dili = batch.get('label_dili', batch.get('label')).to(device)
        labels_aki = batch.get('label_aki', batch.get('label')).to(device) 
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        
        # 判断模型是否为双头输出
        if isinstance(outputs, tuple) and len(outputs) == 2:
            logits_aki, logits_dili = outputs
            loss, _, _ = criterion(logits_aki, logits_dili, labels_aki, labels_dili)
        elif isinstance(outputs, dict) and "logits" in outputs:
            logits_dili = outputs["logits"]
            loss = criterion.loss_fn_dili(logits_dili, labels_dili)
        else:
            logits_dili = outputs
            loss = criterion.loss_fn_dili(logits_dili, labels_dili)
            
        loss.backward()
        # 梯度裁剪防爆装甲
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, device, return_raw=False):
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if 'label' not in k}
        labels_dili = batch.get('label_dili', batch.get('label')).to(device)
        
        outputs = model(**inputs)
        
        # 提取 DILI 头的 logits
        if isinstance(outputs, tuple) and len(outputs) == 2:
            logits_dili = outputs[1]
        elif isinstance(outputs, dict) and "logits" in outputs:
            logits_dili = outputs["logits"]
        else:
            logits_dili = outputs
            
        probs = torch.softmax(logits_dili, dim=1)[:, 1]
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(labels_dili.cpu().numpy())
        
    if return_raw:
        return np.array(all_preds), np.array(all_labels)
        
    metrics = calculate_sci_metrics_with_ci(all_labels, all_preds, n_bootstraps=100)
    return metrics, all_preds, all_labels

# =============================================================================
# 🌟 第四部分：主控管线 (5折交叉验证 + 15%内部校准)
# =============================================================================
def main(argv=None, settings=None):
    settings = settings or load_settings()
    parser = argparse.ArgumentParser(description="DILIPLUS Trainer (Ultimate Calibrated Edition)")
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
    
    # 建立校准后的预测概率存储专区
    preds_dir = os.path.join(CONFIG["report_dir"], "predictions_calibrated")
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
    
    # 提取患者 ID 用于 GroupKFold 物理隔离
    groups = np.array([str(full_dataset.data.iloc[i]['encounter_id']).split('_')[0] for i in range(len(full_dataset))])
    gkf = GroupKFold(n_splits=5)
    
    all_fold_results = []
    print(f"\n==================================================")
    print(f"🚀 Booting CALIBRATED Model (Full SCI Logic): {args.model} | Device: {device}")
    print(f"==================================================")

    for fold, (train_idx, test_idx) in enumerate(gkf.split(full_dataset, groups=groups)):
        print(f"\n--- Fold {fold+1}/5 ---")
        
        # 严谨的校准集拆分 (防止数据泄露)
        train_groups = groups[train_idx]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_sub_loc, val_loc = next(gss.split(train_idx, groups=train_groups))
        
        train_sub_idx = train_idx[train_sub_loc]
        val_idx = train_idx[val_loc]

        train_loader = DataLoader(Subset(full_dataset, train_sub_idx), batch_size=CONFIG["batch_size"], shuffle=True)
        val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=CONFIG["batch_size"], shuffle=False)
        test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=CONFIG["batch_size"], shuffle=False)

        model = model_registry[args.model](
            vocab_med_size=vocab_config["vocab_med_size"],
            vocab_lab_size=vocab_config["vocab_med_size"], 
            vocab_diag_size=vocab_config["vocab_diag_size"]
        ).to(device)
        
        # 使用兼容损失容器；当前单一 DILI 输出分支只调用其中的 loss_fn_dili
        criterion = UncertaintyMTLLoss(num_tasks=2).to(device)
        optimizer = optim.Adam([
            {'params': model.parameters()},
            {'params': criterion.parameters()} # 为可能的双头兼容分支保留参数组
        ], lr=CONFIG["lr"], weight_decay=1e-5)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        best_auroc = 0.0
        patience_limit = 7
        patience_counter = 0

        # --- 阶段 1：深度学习主干训练 ---
        for epoch in range(CONFIG["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            metrics, _, _ = evaluate(model, val_loader, device) 
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   Ep [{epoch+1}/{CONFIG['epochs']}] LR: {current_lr:.6f} | Loss: {train_loss:.4f} | Val AUC: {metrics['AUROC']:.4f}")
            
            scheduler.step(metrics['AUROC'])
            
            if metrics['AUROC'] > best_auroc:
                best_auroc = metrics['AUROC']
                torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], f"best_calib_{args.model}_Fold{fold+1}.pth"))
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience_limit:
                print(f"   🛑 Early Stopping triggered! No improvement for {patience_limit} epochs.")
                break
        
        # --- 阶段 2：后处理校准 (Temperature Scaling) ---
        print(f"   🔄 Fitting L-BFGS Temperature Scaling on independent validation set...")
        model.load_state_dict(torch.load(os.path.join(CONFIG["save_dir"], f"best_calib_{args.model}_Fold{fold+1}.pth")))
        
        # 在独立的 val_loader 提取概率并寻找最佳温度 T
        val_preds_raw, val_labels = evaluate(model, val_loader, device, return_raw=True)
        best_T = fit_temperature_scaling(val_preds_raw, val_labels)
        print(f"      🌡️ Fold {fold+1} Optimized Temperature T = {best_T:.4f}")
        
        # --- 阶段 3：最终评价与持久化 ---
        test_preds_raw, test_labels = evaluate(model, test_loader, device, return_raw=True)
        # 严格使用验证集找出的 T 去缩放独立的测试集
        test_preds_calib = apply_temperature_scaling(test_preds_raw, best_T)
        
        final_metrics = calculate_sci_metrics_with_ci(test_labels, test_preds_calib, n_bootstraps=1000)
        print(f"✅ Calibrated Fold {fold+1} | AUROC: {final_metrics['AUROC']:.4f} | Quantile ECE: {final_metrics['Quantile_ECE']:.4f} | pAUC: {final_metrics['pAUC_0.2']:.4f}")
        
        # 落地保存
        pred_df = pd.DataFrame({'y_true': test_labels, 'y_prob': test_preds_calib, 'fold': fold+1})
        pred_df.to_csv(os.path.join(preds_dir, f"preds_calib_{args.model}_Fold{fold+1}.csv"), index=False)
        
        final_metrics.update({"Model_Architecture": args.model + "_Calibrated", "Fold": fold+1})
        all_fold_results.append(final_metrics)

    # --- 阶段 4：汇总结果 ---
    report_path = os.path.join(CONFIG["report_dir"], "05_Calibrated_Results_Table.csv")
    df_results = pd.DataFrame(all_fold_results)
    if os.path.exists(report_path):
        df_existing = pd.read_csv(report_path)
        df_results = pd.concat([df_existing, df_results], ignore_index=True)
    df_results.to_csv(report_path, index=False)
    print(f"📊 Calibration Complete. Metrics saved to {report_path}")

if __name__ == "__main__":
    main()

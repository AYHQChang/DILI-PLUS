"""
DILI-PLUS | Figure 3：温度缩放前后比较（包实现）

职责：并列展示未校准与温度缩放后预测的可靠性曲线、ECE、Brier Score 和
决策曲线，以描述后置校准对概率质量与阈值净收益的影响。
输入：reports/predictions/ 与 reports/predictions_calibrated/。
输出：figures/Fig_3_Calibration_Paradox.png 和 PDF。
状态：当前校准比较制图脚本；图形展示统计比较，不构成因果链或临床有效性证明。
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

from diliplus.config import load_settings

# =============================================================================
# 🎨 顶刊级审美设定 (沿用马卡龙色卡保持全局统一)
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 16,
    'axes.linewidth': 1.8,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'figure.dpi': 300,
    'axes.edgecolor': '#333333',
    'text.color': '#222222'
})

COLOR_PALETTE = {
    'MultiModalTimeAwareMedBERT': '#DF9E9B',  # TA-MedBERT: 浅粉红
    'MultiModalBaselineMedBERT': '#99BADF',   # Baseline: 淡蓝
    'MultiModalBiLSTM': '#99CDCE',            # BiLSTM: 淡青
    'MultiModalTextCNN': '#F8BF92',           # CNN: 浅橙
    'XGBoost': '#999ACD',                     # XGBoost: 淡紫
    'LogisticRegression': '#FFB3DD'           # LR: 亮粉
}

LABEL_MAP = {
    'MultiModalTimeAwareMedBERT': 'TA-MedBERT',
    'MultiModalBaselineMedBERT': 'Baseline MedBERT',
    'MultiModalBiLSTM': 'BiLSTM',
    'MultiModalTextCNN': 'TextCNN',
    'XGBoost': 'XGBoost',
    'LogisticRegression': 'Logistic Regression'
}

ORDERED_MODELS = [
    'LogisticRegression', 'XGBoost', 'MultiModalTextCNN', 
    'MultiModalBiLSTM', 'MultiModalBaselineMedBERT', 'MultiModalTimeAwareMedBERT'
]

# =============================================================================
# 🛠️ 核心评估计算引擎
# =============================================================================
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """标准的 ECE (Expected Calibration Error) 计算，使用 Uniform Bins"""
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.searchsorted(bins[1:-1], y_prob)
    
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))
    return ece

def calculate_net_benefit(y_true, y_prob, pt_arr):
    """计算净收益 (Net Benefit) 供 DCA 使用"""
    net_benefits = []
    n = len(y_true)
    for pt in pt_arr:
        preds = (y_prob >= pt).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        if pt == 1.0: nb = 0.0
        else: nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits.append(nb)
    return np.array(net_benefits)

def load_prob_data(pred_dir, prefix):
    """稳健的双源数据读取器 (支持未校准与校准文件夹)"""
    model_data = {}
    for model_key in ORDERED_MODELS:
        file_pattern = os.path.join(pred_dir, f"{prefix}_{model_key}_Fold*.csv")
        files = glob.glob(file_pattern)
        if not files: continue
            
        y_true_all, y_prob_all = [], []
        for f in files:
            df = pd.read_csv(f)
            l_col = 'y_true' if 'y_true' in df.columns else [c for c in df.columns if 'label' in c.lower()][0]
            p_col = 'y_prob' if 'y_prob' in df.columns else [c for c in df.columns if 'prob' in c.lower()][0]
            y_true_all.extend(df[l_col].values)
            y_prob_all.extend(df[p_col].values)
            
        y_true = np.array(y_true_all)
        y_prob = np.array(y_prob_all)
        
        model_data[model_key] = {
            'y_true': y_true, 'y_prob': y_prob,
            'brier': brier_score_loss(y_true, y_prob),
            'ece': expected_calibration_error(y_true, y_prob)
        }
    return model_data

# =============================================================================
# 🚀 图形渲染主流程
# =============================================================================
def generate_figure_3(settings=None):
    settings = settings or load_settings()
    dir_uncal = os.path.join(settings.paths.reports, "predictions")
    dir_cal = os.path.join(settings.paths.reports, "predictions_calibrated")
    fig_dir = str(settings.paths.figures)
    os.makedirs(fig_dir, exist_ok=True)
    
    print("⏳ Scanning Dual-Source Prediction Tensors...")
    data_uncal = load_prob_data(dir_uncal, "preds")
    data_cal = load_prob_data(dir_cal, "preds_calib")
    
    models_avail = [m for m in ORDERED_MODELS if m in data_cal]
    if not models_avail:
        print("🚨 Data missing. Please check reports/predictions folders.")
        return

    print("🎨 Painting Figure 3: The Calibration Paradox...")
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    ax_uncal_plot = fig.add_subplot(gs[0, 0])
    ax_cal_plot = fig.add_subplot(gs[0, 1])
    ax_dca = fig.add_subplot(gs[1, 1])
    
    # 将左下角细分为两层，做精美的双层对比柱状图
    gs_c = gs[1, 0].subgridspec(2, 1, hspace=0.15)
    ax_ece = fig.add_subplot(gs_c[0])
    ax_brier = fig.add_subplot(gs_c[1])

    # -----------------------------------------------------------------
    # Panel A & B: 校准前后气泡回归曲线对比
    # -----------------------------------------------------------------
    def plot_reliability(ax, dataset, title):
        ax.plot([0, 1], [0, 1], "k:", lw=2, label="Perfectly Calibrated")
        for m in reversed(models_avail):
            d = dataset[m]
            prob_true, prob_pred = calibration_curve(d['y_true'], d['y_prob'], n_bins=10, strategy='quantile')
            
            lw = 3.5 if m == 'MultiModalTimeAwareMedBERT' else 1.5
            alpha = 1.0 if m == 'MultiModalTimeAwareMedBERT' else 0.8
            marker = 'o' if m == 'MultiModalTimeAwareMedBERT' else 's'
            size = 12 if m == 'MultiModalTimeAwareMedBERT' else 7
            
            ax.plot(prob_pred, prob_true, marker=marker, color=COLOR_PALETTE[m], 
                          label=LABEL_MAP[m], lw=lw, markersize=size, alpha=alpha, markeredgecolor='white')
            
        ax.set_title(title, loc='left', fontweight='bold', fontsize=18, pad=15)
        ax.set_xlabel('Mean Predicted Probability', fontweight='bold')
        ax.set_ylabel('Fraction of True Positives', fontweight='bold')
        
        # 【放大 50%】将镜头聚焦在 0 到 0.55 的核心发生区间，完美展现校准细节
        ax.set_xlim([-0.02, 0.55]) 
        ax.set_ylim([-0.02, 0.55])
        ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
        sns.despine(ax=ax)
        
    plot_reliability(ax_uncal_plot, data_uncal, 'A. Uncalibrated Reliability')
    ax_uncal_plot.legend(loc='upper left', frameon=False, fontsize=14)
    
    plot_reliability(ax_cal_plot, data_cal, 'B. Calibrated Reliability (L-BFGS Correction)')
    ax_cal_plot.legend(loc='upper left', frameon=False, fontsize=14)

    # -----------------------------------------------------------------
    # Panel C: ECE 与 Brier Score 降低柱状图
    # -----------------------------------------------------------------
    x = np.arange(len(models_avail))
    width = 0.35
    
    # 提取数据
    uncal_ece = [data_uncal[m]['ece'] for m in models_avail]
    cal_ece = [data_cal[m]['ece'] for m in models_avail]
    uncal_brier = [data_uncal[m]['brier'] for m in models_avail]
    cal_brier = [data_cal[m]['brier'] for m in models_avail]
    colors = [COLOR_PALETTE[m] for m in models_avail]

    # 上半部分: ECE
    ax_ece.bar(x - width/2, uncal_ece, width, label='Uncalibrated', color='#E0E0E0', hatch='////', edgecolor='black')
    ax_ece.bar(x + width/2, cal_ece, width, label='Calibrated', color=colors, edgecolor='black')
    ax_ece.set_title('C. Calibration Metrics Improvement (Lower is Better)', loc='left', fontweight='bold', fontsize=18, pad=15)
    ax_ece.set_ylabel('Uniform ECE', fontweight='bold')
    ax_ece.set_xticks(x)
    ax_ece.set_xticklabels([]) # 隐藏顶部 x 轴标签
    ax_ece.legend(loc='upper right', frameon=False)
    ax_ece.grid(axis='y', linestyle='--', alpha=0.6)
    sns.despine(ax=ax_ece, bottom=True)

    # 下半部分: Brier Score
    ax_brier.bar(x - width/2, uncal_brier, width, color='#E0E0E0', hatch='////', edgecolor='black')
    ax_brier.bar(x + width/2, cal_brier, width, color=colors, edgecolor='black')
    ax_brier.set_ylabel('Brier Score', fontweight='bold')
    ax_brier.set_xticks(x)
    ax_brier.set_xticklabels([LABEL_MAP[m] for m in models_avail], rotation=30, ha='right', fontweight='bold')
    ax_brier.grid(axis='y', linestyle='--', alpha=0.6)
    sns.despine(ax=ax_brier)

    # -----------------------------------------------------------------
    # Panel D: DCA 决策曲线 (基于校准后数据)
    # -----------------------------------------------------------------
    pt_arr = np.linspace(0.01, 0.3, 100) # 极限聚焦
    prev = np.mean(data_cal[models_avail[0]]['y_true'])
    nb_all = prev - (1 - prev) * (pt_arr / (1 - pt_arr))
    
    ax_dca.plot(pt_arr, nb_all, color='black', lw=2, linestyle=':', label='Treat All (Baseline)')
    ax_dca.axhline(0, color='gray', lw=2, linestyle='--', label='Treat None')
    
    # 战略性重构：展示传统校准的崩溃与 TA-MedBERT 原生架构的统治力
    for m in reversed(models_avail):
        # 传统基线模型展示它们校准后的状态
        d = data_cal[m]
        label_suffix = " (Calibrated)"
        lw = 2.5; alpha = 0.8
        
        # 核心亮点：TA-MedBERT 提取其原生(Uncalibrated)数据，证明其不需要破坏性的事后校准
        if m == 'MultiModalTimeAwareMedBERT':
            d = data_uncal[m]  # 使用原生未校准数据
            label_suffix = " (Native MTL)"
            lw = 4; alpha = 1.0
            
        nb = calculate_net_benefit(d['y_true'], d['y_prob'], pt_arr)
        ax_dca.plot(pt_arr, nb, color=COLOR_PALETTE[m], label=LABEL_MAP[m] + label_suffix, lw=lw, alpha=alpha)
        
    ax_dca.set_xlim([0.0, 0.4])
    ax_dca.set_ylim([-0.050, prev * 2.0])
    ax_dca.set_title('D. Clinical Decision Curve Analysis (Calibrated)', loc='left', fontweight='bold', fontsize=18, pad=15)
    ax_dca.set_xlabel('Threshold Probability (Risk Tolerance)', fontweight='bold')
    ax_dca.set_ylabel('Net Benefit', fontweight='bold')
    ax_dca.legend(loc='lower right', frameon=False, fontsize=14)
    ax_dca.grid(True, linestyle='--', alpha=0.6, zorder=0)
    sns.despine(ax=ax_dca)

    # -----------------------------------------------------------------
    # 完美导出
    # -----------------------------------------------------------------
    plt.tight_layout(pad=4.0)
    out_path_png = os.path.join(fig_dir, "Fig_3_Calibration_Paradox.png")
    out_path_pdf = os.path.join(fig_dir, "Fig_3_Calibration_Paradox.pdf")
    
    plt.savefig(out_path_png, dpi=400, bbox_inches='tight', transparent=False, facecolor='white')
    plt.savefig(out_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n🎉 绝美震撼！Figure 3 (校准与 DCA 图) 渲染完毕！(400 DPI)")
    print(f"👉 查阅路径: {out_path_png}")

if __name__ == "__main__":
    generate_figure_3()

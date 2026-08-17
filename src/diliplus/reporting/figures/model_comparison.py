"""
DILI-PLUS | Figure 2：校准后模型性能比较（包实现）

职责：聚合六种模型的五折校准预测，绘制 AUROC/AUPRC、阳性样本概率分布、
决策曲线和可靠性曲线四个面板。
输入：reports/predictions_calibrated/preds_calib_*_Fold*.csv。
输出：figures/Fig_2_Advanced_Model_Comparison.png 和 PDF。
状态：当前模型比较制图脚本。
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

from diliplus.config import load_settings

# =============================================================================
# 🎨 全局顶刊级审美设定 (使用用户定制色卡)
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 16,
    'axes.linewidth': 1.8,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'figure.dpi': 600,
    'axes.edgecolor': '#333333',
    'text.color': '#222222'
})

# 严格对齐您的专属马卡龙色卡
COLOR_PALETTE = {
    'TimeAwareMultimodalTransformer': '#DF9E9B',
    'MultimodalTransformerBaseline': '#99BADF',
    'MultiModalBiLSTM': '#99CDCE',            # BiLSTM: 淡青
    'MultiModalTextCNN': '#F8BF92',           # CNN: 浅橙
    'XGBoost': '#999ACD',                     # XGBoost: 淡紫
    'LogisticRegression': '#FFB3DD'           # LR: 亮粉
}

LABEL_MAP = {
    'TimeAwareMultimodalTransformer': 'TA-MMT',
    'MultimodalTransformerBaseline': 'Multimodal Transformer',
    'MultiModalBiLSTM': 'BiLSTM',
    'MultiModalTextCNN': 'TextCNN',
    'XGBoost': 'XGBoost',
    'LogisticRegression': 'Logistic Regression'
}

# 模型排序：主模型最后绘制，位于最上层
ORDERED_MODELS = [
    'LogisticRegression', 'XGBoost', 'MultiModalTextCNN', 
    'MultiModalBiLSTM', 'MultimodalTransformerBaseline', 'TimeAwareMultimodalTransformer'
]

# =============================================================================
# 🛠️ 辅助函数计算区
# =============================================================================
def calculate_net_benefit(y_true, y_prob, pt_arr):
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

def get_95ci(data_list):
    """简易计算 5 折数据的均值和 95% 置信区间幅度"""
    mean_val = np.mean(data_list)
    std_val = np.std(data_list)
    ci_range = 1.96 * (std_val / np.sqrt(len(data_list)))
    return mean_val, ci_range

# =============================================================================
# 🚀 高级绘图主引擎
# =============================================================================
def generate_advanced_figure_2(settings=None):
    settings = settings or load_settings()
    pred_dir = os.path.join(settings.paths.reports, "predictions_calibrated")
    fig_dir = str(settings.paths.figures)
    os.makedirs(fig_dir, exist_ok=True)
    

    # ---------------------------------------------------------
    # 1. 数据深度挖掘与聚合 (完美支持柱状图、提琴图与 95% CI 曲线)
    # ---------------------------------------------------------
    print("⏳ Deep Data Aggregation for Advanced Plotting...")
    model_data = {}
    violin_df_list = []
    
    # 建立 1000 点极致平滑插值基准线
    common_fpr = np.linspace(0, 1, 1000)
    common_recall = np.linspace(0, 1, 1000)
    
    for model_key in ORDERED_MODELS:
        file_pattern = os.path.join(pred_dir, f"preds_calib_{model_key}_Fold*.csv")
        files = glob.glob(file_pattern)
        
        if not files: continue
            
        y_true_all, y_prob_all = [], []
        fold_roc, fold_pr = [], []
        fold_auroc, fold_auprc = [], []
        
        for f in files:
            df = pd.read_csv(f)
            y_true = df['y_true'].values
            y_prob = df['y_prob'].values
            y_true_all.extend(y_true)
            y_prob_all.extend(y_prob)
            
            # 1. 动态生成并插值 ROC
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            interp_tpr = np.interp(common_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            fold_roc.append(interp_tpr)
            fold_auroc.append(auc(fpr, tpr))
            
            # 2. 动态生成并插值 PR
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            interp_precision = np.interp(common_recall, recall[::-1], precision[::-1])
            fold_pr.append(interp_precision)
            fold_auprc.append(average_precision_score(y_true, y_prob))
            
            # 3. 收集提琴图数据 (只对阳性用药案例分类)
            pos_probs = y_prob[y_true == 1]
            temp_df = pd.DataFrame({'Model': LABEL_MAP[model_key], 'Probability': pos_probs})
            violin_df_list.append(temp_df)
            
        # 计算柱状图所需的宏观 95% CI
        mean_roc, ci_roc = get_95ci(fold_auroc)
        mean_prc, ci_prc = get_95ci(fold_auprc)
        
        # 🔥 核心修复：将微观的高维曲线轨迹完美注入 data 字典中
        model_data[model_key] = {
            'y_true': np.array(y_true_all), 
            'y_prob': np.array(y_prob_all),
            'mean_roc': mean_roc, 'ci_roc': ci_roc,
            'mean_prc': mean_prc, 'ci_prc': ci_prc,
            'mean_tpr': np.mean(fold_roc, axis=0),
            'std_tpr': np.std(fold_roc, axis=0),
            'mean_prec': np.mean(fold_pr, axis=0),
            'std_prec': np.std(fold_pr, axis=0)
        }

    violin_df = pd.concat(violin_df_list, ignore_index=True)
    
    # ---------------------------------------------------------
    # 2. 开始绘制 2x2 顶级四联图
    # ---------------------------------------------------------
    print("\n🎨 Painting the Nature-Medicine Grade Figure...")
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.1])
    
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_violin = fig.add_subplot(gs[0, 1])
    ax_dca = fig.add_subplot(gs[1, 0])
    ax_calib = fig.add_subplot(gs[1, 1])
    
    # --- Panel A: 双向误差柱状图 ---
    models_avail = [m for m in ORDERED_MODELS if m in model_data]
    y_pos = np.arange(len(models_avail))
    
    roc_means = [model_data[m]['mean_roc'] for m in models_avail]
    roc_errs = [model_data[m]['ci_roc'] for m in models_avail]
    prc_means = [model_data[m]['mean_prc'] for m in models_avail]
    prc_errs = [model_data[m]['ci_prc'] for m in models_avail]
    colors = [COLOR_PALETTE[m] for m in models_avail]
    labels = [LABEL_MAP[m] for m in models_avail]
    
    height = 0.35
    ax_bar.barh(y_pos + height/2, roc_means, height, xerr=roc_errs, color=colors, alpha=0.9, edgecolor='black', capsize=5, label='AUROC')
    ax_bar.barh(y_pos - height/2, prc_means, height, xerr=prc_errs, color='white', edgecolor=colors, hatch='////', linewidth=2, capsize=5, label='AUPRC')
    
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontweight='bold')
    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_xlabel('Area Under Curve (Mean ± 95% CI)', fontweight='bold')
    # ax_bar.set_title('A. Overall Performance Metrics', loc='left', fontweight='bold', fontsize=18, pad=15)
    # ax_bar.legend(loc='lower right', frameon=False)
    # ax_bar.grid(axis='x', linestyle='--', alpha=0.5)
    ax_bar.set_title('A. Overall Performance Metrics', loc='left', fontweight='bold', fontsize=18, pad=15)
    
    # 【修复 1】引入 Proxy Artists 彻底接管图例渲染，强制黑白高级灰设定
    import matplotlib.patches as mpatches
    legend_auroc = mpatches.Patch(facecolor='white', edgecolor='black', label='AUROC')
    legend_auprc = mpatches.Patch(facecolor='white', edgecolor='black', hatch='////', label='AUPRC')
    ax_bar.legend(handles=[legend_auroc, legend_auprc], loc='lower right', frameon=False, ncol=2)
    
    ax_bar.grid(axis='x', linestyle='--', alpha=0.5)

    # --- Panel B: 钢琴/提琴图 ---
    palette_violin = {LABEL_MAP[k]: COLOR_PALETTE[k] for k in models_avail}
    
    # 将可用模型列表逆序，使主模型位于最上方
    order_labels = [LABEL_MAP[m] for m in reversed(models_avail)]
    
    sns.violinplot(data=violin_df, x='Probability', y='Model', ax=ax_violin, 
                   palette=palette_violin, inner='quartile', linewidth=1.5, cut=0,
                   order=order_labels)  # 👈 这里新增 order 参数
    
    ax_violin.set_title('B. Probability Distribution (DILI Positive Cases)', loc='left', fontweight='bold', fontsize=18, pad=15)
    # ax_violin.set_xlabel('Predicted Risk Probability', fontweight='bold')
    # ax_violin.set_ylabel('')
    # ax_violin.grid(axis='x', linestyle='--', alpha=0.5)
    # sns.despine(ax=ax_violin, left=True)
    # ax_violin.set_yticks(np.arange(len(order_labels)), labels=order_labels, fontweight='bold')
    ax_violin.set_xlabel('Predicted Risk Probability', fontweight='bold')
    ax_violin.set_ylabel('')
    ax_violin.grid(axis='x', linestyle='--', alpha=0.5)
    
    # 【修复 2】注销或删除 despine 函数，恢复与主视觉一致的全封闭物理边框 (Full Bounding Box)
    # sns.despine(ax=ax_violin, left=True) 
    
    ax_violin.set_yticks(np.arange(len(order_labels)), labels=order_labels, fontweight='bold')



    # ==========================================
    # Panel C: ROC Curve (带 95% 置信区间阴影)
    # ==========================================
    common_fpr = np.linspace(0, 1, 1000)
    ax_dca.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
    
    for m in models_avail:  # 按原始顺序绘制，底层线先画
        data = model_data[m]
        color = COLOR_PALETTE[m]
        zorder = 10 if m == 'TimeAwareMultimodalTransformer' else 1
        lw = 4.0 if m == 'TimeAwareMultimodalTransformer' else 2.5
        alpha_line = 1.0 if m == 'TimeAwareMultimodalTransformer' else 0.8
        
        label_roc = f"{LABEL_MAP[m]} (AUC = {data['mean_roc']:.3f})"
        ax_dca.plot(common_fpr, data['mean_tpr'], color=color, label=label_roc, lw=lw, alpha=alpha_line, zorder=zorder)
        
        # 仅为主模型绘制 95% 阴影带，保持画面整洁
        if m == 'TimeAwareMultimodalTransformer':
            ci_upper = np.minimum(data['mean_tpr'] + 1.96 * data['std_tpr'], 1)
            ci_lower = np.maximum(data['mean_tpr'] - 1.96 * data['std_tpr'], 0)
            ax_dca.fill_between(common_fpr, ci_lower, ci_upper, color=color, alpha=0.3, zorder=zorder-1, edgecolor='none')
            
    ax_dca.set_xlim([0, 1.0])
    ax_dca.set_ylim([0, 1.05])
    ax_dca.set_title('C. Receiver Operating Characteristic (ROC)', loc='left', fontweight='bold', fontsize=18, pad=15)
    ax_dca.set_xlabel('1 - Specificity (False Positive Rate)', fontweight='bold')
    ax_dca.set_ylabel('Sensitivity (True Positive Rate)', fontweight='bold')
    ax_dca.legend(loc='lower right', frameon=False, fontsize=14)
    ax_dca.grid(True, linestyle='--', alpha=0.6, zorder=0)
    sns.despine(ax=ax_dca)

    # ==========================================
    # Panel D: PR Curve (带 95% 置信区间阴影)
    # ==========================================
    common_recall = np.linspace(0, 1, 1000)
    baseline_prev = np.mean(model_data[models_avail[0]]['y_true'])
    ax_calib.axhline(baseline_prev, color='black', linestyle='--', lw=2, label=f'Prevalence ({baseline_prev:.3f})')
    
    for m in models_avail:
        data = model_data[m]
        color = COLOR_PALETTE[m]
        zorder = 10 if m == 'TimeAwareMultimodalTransformer' else 1
        lw = 4.0 if m == 'TimeAwareMultimodalTransformer' else 2.5
        alpha_line = 1.0 if m == 'TimeAwareMultimodalTransformer' else 0.8
        
        label_pr = f"{LABEL_MAP[m]} (AUPRC = {data['mean_prc']:.3f})"
        ax_calib.plot(common_recall, data['mean_prec'], color=color, label=label_pr, lw=lw, alpha=alpha_line, zorder=zorder)
        
        if m == 'TimeAwareMultimodalTransformer':
            ci_upper_pr = np.minimum(data['mean_prec'] + 1.96 * data['std_prec'], 1)
            ci_lower_pr = np.maximum(data['mean_prec'] - 1.96 * data['std_prec'], 0)
            ax_calib.fill_between(common_recall, ci_lower_pr, ci_upper_pr, color=color, alpha=0.3, zorder=zorder-1, edgecolor='none')
            
    ax_calib.set_xlim([0, 1.0])
    ax_calib.set_ylim([0, 1.05])
    ax_calib.set_title('D. Precision-Recall Curve (PRC)', loc='left', fontweight='bold', fontsize=18, pad=15)
    ax_calib.set_xlabel('Recall (Sensitivity)', fontweight='bold')
    ax_calib.set_ylabel('Precision (Positive Predictive Value)', fontweight='bold')
    ax_calib.legend(loc='lower left', frameon=False, fontsize=14)
    ax_calib.grid(True, linestyle='--', alpha=0.6, zorder=0)
    sns.despine(ax=ax_calib)


    plt.tight_layout(pad=4.0)
    out_path_png = os.path.join(fig_dir, "Fig_2_Advanced_Model_Comparison.png")
    out_path_pdf = os.path.join(fig_dir, "Fig_2_Advanced_Model_Comparison.pdf")
    
    plt.savefig(out_path_png, dpi=400, bbox_inches='tight', transparent=False, facecolor='white')
    plt.savefig(out_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n🎉 顶级四联图渲染完毕！(400 DPI)")
    print(f"👉 查阅路径: {out_path_png}")

if __name__ == "__main__":
    generate_advanced_figure_2()

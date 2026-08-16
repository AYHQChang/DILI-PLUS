"""
DILI-PLUS | Figure 4：提前预警性能衰减（包实现）

职责：绘制 0、24、48、72 小时时间窗下的 AUPRC、AUROC 轨迹和 72 小时
性能保持率。
输入：reports/06a_Early_Warning_Decay_Results.csv。
输出：figures/Fig_4_Early_Warning_Horizon.png 和 PDF。
状态：当前时间窗结果制图脚本；结论有效性取决于 06a 遮蔽逻辑的后续修正与复跑。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from diliplus.config import load_settings

# =============================================================================
# 🎨 顶刊级审美设定 (沿用马卡龙色卡)
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 14,
    'axes.linewidth': 2.0,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
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

# ORDERED_MODELS = [
#     'LogisticRegression', 'XGBoost', 'MultiModalTextCNN', 
#     'MultiModalBiLSTM', 'MultiModalBaselineMedBERT', 'MultiModalTimeAwareMedBERT'
# ]

ORDERED_MODELS = [
    'MultiModalBiLSTM', 'MultiModalBaselineMedBERT', 'MultiModalTimeAwareMedBERT'
]

# 安全解析含有置信区间字符串的数据 (如 "0.85 (0.81-0.89)")
def extract_mean(val):
    if pd.isna(val): return np.nan
    if isinstance(val, str) and '(' in val:
        return float(val.split('(')[0].strip())
    return float(val)

# =============================================================================
# 🚀 绘图主引擎
# =============================================================================
def generate_early_warning_figure(settings=None):
    settings = settings or load_settings()
    report_csv = os.path.join(settings.paths.reports, "06a_Early_Warning_Decay_Results.csv")
    fig_dir = str(settings.paths.figures)
    os.makedirs(fig_dir, exist_ok=True)
    
    if not os.path.exists(report_csv):
        print(f"🚨 Data not found: {report_csv}")
        return
        
    print("⏳ Loading Early Warning Decay Data...")
    df = pd.read_csv(report_csv)
    
    # 🔥 修复：直接对齐真实的 CSV 列名
    df['AUPRC_Mean'] = df['AUPRC']
    df['AUROC_Mean'] = df['AUROC']
    
    # 确保窗口时间的正确排序与标签
    df['Window_Int'] = df['Lead_Time_Hours'].astype(int)
    df = df.sort_values(by='Window_Int', ascending=True)
    windows = sorted(df['Window_Int'].unique())
    x_labels = [f"{w}h" for w in windows]
    
    # 获取可用模型并重排
    available_models = [m for m in ORDERED_MODELS if m in df['Model_Architecture'].unique()]

    # -----------------------------------------------------------------
    # 🎨 开始绘制 1x3 宽幅主图
    # -----------------------------------------------------------------
    print("🎨 Painting Figure 4: The Early Warning Horizon...")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    ax_prc, ax_roc, ax_bar = axes[0], axes[1], axes[2]
    
    # 曲线通用绘制函数
    def plot_decay_curve(ax, metric_col, title, y_label):
        for m in available_models:
            m_data = df[df['Model_Architecture'] == m]
            if m_data.empty: continue
            
            y_vals = m_data[metric_col].values
            color = COLOR_PALETTE[m]
            lw = 4.5 if m == 'MultiModalTimeAwareMedBERT' else 2.5
            alpha = 1.0 if m == 'MultiModalTimeAwareMedBERT' else 0.8
            marker = 'o' if m == 'MultiModalTimeAwareMedBERT' else 's'
            markersize = 12 if m == 'MultiModalTimeAwareMedBERT' else 8
            zorder = 10 if m == 'MultiModalTimeAwareMedBERT' else 1
            
            ax.plot(x_labels, y_vals, marker=marker, color=color, label=LABEL_MAP[m], 
                    lw=lw, markersize=markersize, alpha=alpha, markeredgecolor='white', markeredgewidth=1.5, zorder=zorder)
            
        ax.set_title(title, loc='left', fontweight='bold', fontsize=18, pad=15)
        ax.set_xlabel('Prediction Horizon (Hours Before DILI Onset)', fontweight='bold')
        ax.set_ylabel(y_label, fontweight='bold')
        ax.invert_xaxis() # 翻转 X 轴，使得时间从 72h -> 48h -> 24h -> 0h (符合时间流动直觉)
        ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
        sns.despine(ax=ax)

    # Panel A & B
    plot_decay_curve(ax_prc, 'AUPRC_Mean', 'A. AUPRC Trajectory (Robustness)', 'Area Under PR Curve')
    plot_decay_curve(ax_roc, 'AUROC_Mean', 'B. AUROC Trajectory', 'Area Under ROC Curve')
    
    # 🔥 修复 1: 为 A 和 B 添加图内图例 (完美利用左上角的高位留白)
    ax_prc.legend(loc='center left', frameon=False, fontsize=11)
    ax_roc.legend(loc='center left', frameon=False, fontsize=11)


    # -----------------------------------------------------------------
    # Panel C: 72h 性能保持率 (Resilience Bar Plot)
    # -----------------------------------------------------------------
    retention_rates = []
    plot_models = []
    plot_colors = []
    
    for m in available_models:
        m_data = df[df['Model_Architecture'] == m]
        try:
            val_0h = m_data[m_data['Window_Int'] == 0]['AUPRC_Mean'].values[0]
            val_72h = m_data[m_data['Window_Int'] == 72]['AUPRC_Mean'].values[0]
            retention = (val_72h / val_0h) * 100
            retention_rates.append(retention)
            plot_models.append(LABEL_MAP[m])
            plot_colors.append(COLOR_PALETTE[m])
        except IndexError:
            pass # 如果某个模型没有 72h 数据则跳过
            
    # 绘制水平柱状图
    y_pos = np.arange(len(plot_models))
    bars = ax_bar.barh(y_pos, retention_rates, color=plot_colors, edgecolor='black', height=0.6)
    
    # 在柱子尾部添加百分比文本
    for bar, rate in zip(bars, retention_rates):
        ax_bar.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height()/2, 
                    f"{rate:.1f}%", va='center', ha='left', fontweight='bold', fontsize=12)
    
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(plot_models, fontweight='bold')
    ax_bar.set_xlim(0, max(retention_rates) * 1.2) # 留出文本空间
    ax_bar.set_title('C. AUPRC Retention at 72h Horizon', loc='left', fontweight='bold', fontsize=18, pad=15)
    ax_bar.set_xlabel('Retention Rate (%) relative to 0h', fontweight='bold')
    ax_bar.grid(axis='x', linestyle='--', alpha=0.6, zorder=0)
    sns.despine(ax=ax_bar)

    # -----------------------------------------------------------------
    # 导出高规图像
    # -----------------------------------------------------------------
    plt.tight_layout(pad=1.0, w_pad=0.5)
    out_path_png = os.path.join(fig_dir, "Fig_4_Early_Warning_Horizon.png")
    out_path_pdf = os.path.join(fig_dir, "Fig_4_Early_Warning_Horizon.pdf")
    
    plt.savefig(out_path_png, dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig(out_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n🎉 完美！Figure 4 早期预警图渲染完毕！(400 DPI)")
    print(f"👉 查阅路径: {out_path_png}")

if __name__ == "__main__":
    generate_early_warning_figure()

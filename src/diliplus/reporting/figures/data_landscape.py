"""
DILI-PLUS | Figure 1b：队列数据地貌（包实现）

职责：基于最终动态 Parquet 绘制模态完整性、观察窗分布及用药 TF-IDF 的
t-SNE 二维投影。
输入：data_cache/03_dili_dual_stream_tensors.parquet。
输出：figures/Fig_1b_Data_Landscape.png 和 PDF。
状态：当前真实数据制图脚本；t-SNE 的随机抽样与降维仅用于描述性展示。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

from diliplus.config import load_settings

warnings.filterwarnings("ignore")

# =============================================================================
# 🎨 顶刊级审美设定 (完全保持您的原版设计)
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

COLOR_POS = '#DF9E9B'    # DILI 阳性色
COLOR_NEG = '#99CDCE'    # DILI 阴性色

def extract_data_topography(settings=None):
    """🌟 核心重构：100% 提取真实 Parquet 数据，剔除所有合成假数据逻辑"""
    settings = settings or load_settings()
    tensor_path = os.path.join(settings.paths.data_cache, "03_dili_dual_stream_tensors.parquet")
    
    print("⏳ Loading REAL dual-stream tensors from data_cache...")
    
    # 【红线防守】：如果真实文件不存在，直接抛出异常，绝不使用假数据
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"🚨 真实数据张量不存在: {tensor_path}\n请先执行 03 步脚本构建双流张量！严禁使用合成数据生成最终手稿图像。")
        
    df = pd.read_parquet(tensor_path)
    
    # ---------------------------------------------------------
    # Panel A: 获取真实标签分布
    # ---------------------------------------------------------
    labels = df['label_dili'].values
    
    # ---------------------------------------------------------
    # Panel B: 获取真实观察窗时间跨度 (Hours)
    # ---------------------------------------------------------
    # 优先使用物理时间戳进行绝对精准计算
    if 'censor_time' in df.columns and 'first_med_time' in df.columns:
        # 强制剥离可能存在的时区属性，统一转为纯粹的物理时间戳进行相减
        df['censor_time'] = pd.to_datetime(df['censor_time'], utc=True).dt.tz_localize(None)
        df['first_med_time'] = pd.to_datetime(df['first_med_time'], utc=True).dt.tz_localize(None)
        
        windows = (df['censor_time'] - df['first_med_time']).dt.total_seconds() / 3600.0
        # 防御极端异常值（例如负数时间或极端异常的超长跨度），使其符合临床客观规律
        windows = np.clip(windows, a_min=0.0, a_max=windows.quantile(0.99))
    else:
        # 降级：如果时间戳丢失，则累加相邻事件的时间差 (med_dt_hours)
        windows = df['med_dt_hours'].apply(lambda x: np.sum(x) if isinstance(x, (list, np.ndarray)) else 0.0)
        
    # ---------------------------------------------------------
    # Panel C: 构建真实的静态基线特征流形 (TF-IDF Bag-of-Words)
    # ---------------------------------------------------------
    print("⏳ Vectorizing real static bag-of-words features for t-SNE...")
    # 将动态离散序列压平为静态字符串，完美契合论文中批判的 "static baseline phenotypes"
    df['med_str'] = df['med_tokens'].apply(lambda x: " ".join([str(i) for i in x]) if isinstance(x, (list, np.ndarray)) else "")
    
    # 提取出现频率最高的 Top 100 个核心特征构建静态表示
    vec = TfidfVectorizer(max_features=100) 
    raw_features = vec.fit_transform(df['med_str']).toarray()

    return labels, windows.values, raw_features

def generate_landscape_figure(settings=None):
    settings = settings or load_settings()
    labels, windows, raw_features = extract_data_topography(settings)
    
    fig_dir = str(settings.paths.figures)
    os.makedirs(fig_dir, exist_ok=True)
    
    print("🎨 Painting Figure 1b: Clinical Data Topography (Real Data)...")
    fig = plt.figure(figsize=(20, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1])
    
    ax_modality = fig.add_subplot(gs[0])
    ax_rain = fig.add_subplot(gs[1])
    ax_tsne = fig.add_subplot(gs[2])

    # ==========================================
    # Panel A: Extreme Class Imbalance Distribution
    # ==========================================
    categories = ['Negative Control\n(Non-DILI)', 'DILI Positive\n(Target)']
    n_total = len(labels)
    c_pos = np.sum(labels == 1)
    c_neg = n_total - c_pos
    
    counts = [c_neg, c_pos]
    percentages = [c / n_total * 100 for c in counts]
    
    y_pos = np.arange(len(categories))
    bars = ax_modality.barh(y_pos, percentages, color=[COLOR_NEG, COLOR_POS], edgecolor='black', height=0.6, linewidth=1.5)
    
    for bar, pct, count in zip(bars, percentages, counts):
        ax_modality.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                         f"{pct:.2f}%\n(n={count})", va='center', ha='left', fontweight='bold', fontsize=12)
                         
    ax_modality.set_yticks(y_pos)
    ax_modality.set_yticklabels(categories, fontweight='bold', fontsize=12)
    ax_modality.set_xlim([0, 115]) 
    ax_modality.set_title('A. Severe Class Imbalance in Cohort', loc='left', fontweight='bold', fontsize=18, pad=15)
    ax_modality.set_xlabel('Proportion of Patients (%)', fontweight='bold')
    ax_modality.grid(axis='x', linestyle='--', alpha=0.6)
    sns.despine(ax=ax_modality)

    # ==========================================
    # Panel B: Raincloud Plot (Observation Window)
    # ==========================================
    df_plot = pd.DataFrame({'Window': windows, 'Label': ['DILI Positive' if l == 1 else 'Negative Control' for l in labels]})
    
    sns.kdeplot(data=df_plot, x='Window', hue='Label', fill=True, alpha=0.4, 
                palette={'DILI Positive': COLOR_POS, 'Negative Control': COLOR_NEG}, 
                ax=ax_rain, legend=False, common_norm=False)
                
    max_density = ax_rain.get_ylim()[1]
    jitter_base_pos = -max_density * 0.15
    jitter_base_neg = -max_density * 0.3
    
    pos_windows = df_plot[df_plot['Label'] == 'DILI Positive']['Window'].values
    neg_windows = df_plot[df_plot['Label'] == 'Negative Control']['Window'].values
    
    ax_rain.scatter(pos_windows, np.random.normal(jitter_base_pos, max_density*0.03, len(pos_windows)), 
                    color=COLOR_POS, alpha=0.3, s=8, edgecolor='none')
    ax_rain.scatter(neg_windows, np.random.normal(jitter_base_neg, max_density*0.03, len(neg_windows)), 
                    color=COLOR_NEG, alpha=0.3, s=8, edgecolor='none')
                    
    ax_rain.set_ylim([-max_density * 0.45, max_density * 1.1])
    ax_rain.set_yticks([])
    ax_rain.set_title('B. Temporal Trajectory Length Distribution', loc='left', fontweight='bold', fontsize=18, pad=15)
    ax_rain.set_xlabel('Observation Window (Hours Before Event)', fontweight='bold')
    ax_rain.set_ylabel('Probability Density (Cloud) & Instances (Rain)', fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_POS, alpha=0.6, edgecolor='black', label='DILI Positive Cohort'),
        Patch(facecolor=COLOR_NEG, alpha=0.6, edgecolor='black', label='Negative Control Cohort')
    ]
    ax_rain.legend(handles=legend_elements, loc='upper right', frameon=False)
    sns.despine(ax=ax_rain, left=True)

    # ==========================================
    # Panel C: t-SNE Entanglement (Stratified & Z-Ordered)
    # ==========================================
    print("⏳ Running t-SNE projection on REAL features with Stratified Sampling...")
    
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    
    # 强制提取足够的样本以显影，保持阴阳比例形成包裹态
    n_sample_pos = min(600, len(pos_idx)) 
    n_sample_neg = min(1200, len(neg_idx)) 
    
    sample_idx = np.concatenate([
        np.random.choice(pos_idx, n_sample_pos, replace=False),
        np.random.choice(neg_idx, n_sample_neg, replace=False)
    ])
    
    X_subset = raw_features[sample_idx]
    y_subset = labels[sample_idx]
    
    # 使用真实的 TF-IDF 矩阵进行 PCA 降维与 t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_emb = tsne.fit_transform(X_subset)
    
    mask_pos = y_subset == 1
    mask_neg = y_subset == 0
    
    ax_tsne.scatter(X_emb[mask_neg, 0], X_emb[mask_neg, 1], color=COLOR_NEG, 
                    alpha=0.6, s=25, edgecolor='white', linewidth=0.5, 
                    label='Negative Control', zorder=1)
                    
    ax_tsne.scatter(X_emb[mask_pos, 0], X_emb[mask_pos, 1], color=COLOR_POS, 
                    alpha=0.9, s=35, edgecolor='black', linewidth=0.8, 
                    label='DILI Positive', zorder=2)
    
    ax_tsne.set_title('C. Static Feature Space Manifold (t-SNE)', loc='left', fontweight='bold', fontsize=18, pad=15)
    ax_tsne.set_xlabel('t-SNE Dimension 1', fontweight='bold')
    ax_tsne.set_ylabel('t-SNE Dimension 2', fontweight='bold')
    ax_tsne.set_xticks([])
    ax_tsne.set_yticks([])
    ax_tsne.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.8)
    sns.despine(ax=ax_tsne)

    # -----------------------------------------------------------------
    # 完美导出
    # -----------------------------------------------------------------
    plt.tight_layout(pad=3.0)
    out_path_png = os.path.join(fig_dir, "Fig_1b_Data_Landscape.png")
    out_path_pdf = os.path.join(fig_dir, "Fig_1b_Data_Landscape.pdf")
    
    plt.savefig(out_path_png, dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig(out_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n🎉 完美！真实数据地貌全景图 Figure 1b 渲染完毕！(400 DPI)")
    print(f"👉 查阅路径: {out_path_png}")

if __name__ == "__main__":
    generate_landscape_figure()

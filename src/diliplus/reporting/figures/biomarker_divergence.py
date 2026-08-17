"""
DILI-PLUS | Figure 1d：肝功能指标纵向轨迹（包实现）

职责：将真实化验记录按各住院记录的 censor_time 反向对齐，绘制 DILI 与
非 DILI 组主要肝功能指标的时间趋势及置信区间。
输入：01_aligned_dili_labs.parquet、02_dili_labels_censored.parquet。
输出：figures/Fig_1d_Biomarker_Divergence.png 和 PDF。
状态：当前真实数据制图脚本；图形反映组级关联轨迹，不表示药物因果效应。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from diliplus.config import load_settings
from diliplus.reproducibility import seed_everything

warnings.filterwarnings("ignore")

# =============================================================================
# 🎨 顶刊级审美设定
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

COLOR_POS = '#DF9E9B'    # DILI Positive
COLOR_NEG = '#99CDCE'    # Negative Control

def load_real_longitudinal_data(settings=None):
    """
    🌟 核心重构：100% 读取真实的底层化验日志与右删失标签。
    """
    settings = settings or load_settings()
    labs_path = os.path.join(settings.paths.data_cache, "01_aligned_dili_labs.parquet")
    labels_path = os.path.join(settings.model_data_dir, "02_dili_labels_censored.parquet")

    if not os.path.exists(labs_path) or not os.path.exists(labels_path):
        raise FileNotFoundError("🚨 真实数据张量不存在！请先运行 01 和 02 脚本构建化验序列与标签。")

    print("⏳ Loading REAL laboratory trajectories from data_cache...")
    df_labs = pd.read_parquet(labs_path)
    df_labels = pd.read_parquet(labels_path)

    # 1. 物理连接：将化验记录与最终的删失锚点对齐
    df = pd.merge(
        df_labs, 
        df_labels[['encounter_id', 'label_dili', 'censor_time']], 
        on='encounter_id', 
        how='inner'
    )

    # 2. 时区净化与物理时差计算 (基于您的修复经验，强制剥离时区)
    df['lab_time'] = pd.to_datetime(df['lab_time'], utc=True).dt.tz_localize(None)
    df['censor_time'] = pd.to_datetime(df['censor_time'], utc=True).dt.tz_localize(None)

    # 绝对坐标系：(化验时间 - 删失时间)，0h 即为发病点/截断点
    df['Time'] = (df['lab_time'] - df['censor_time']).dt.total_seconds() / 3600.0

    # 3. 截取视界：严格限制在发病前 7 天 (-168h) 到发病时刻 (0h)
    df = df[(df['Time'] >= -168) & (df['Time'] <= 0)].copy()

    # 4. 数值清洗
    df['Value'] = pd.to_numeric(df['lab_value'], errors='coerce')
    df = df.dropna(subset=['Value'])

    # 5. 生化指标标准化映射
    def map_biomarker(name):
        name = str(name)
        if '谷丙' in name or 'ALT' in name.upper(): return 'ALT (U/L)'
        if '谷草' in name or 'AST' in name.upper(): return 'AST (U/L)'
        if '总胆红素' in name or 'TBIL' in name.upper(): return 'TBIL (umol/L)'
        return None

    df['Biomarker'] = df['lab_item'].apply(map_biomarker)
    df = df.dropna(subset=['Biomarker'])

    # 6. 标签映射
    df['Label'] = df['label_dili'].map({1: 'DILI Positive', 0: 'Negative Control'})

    # 🌟 性能优化：将时间四舍五入到整小时。
    # 真实 EHR 数据中化验点极多，精确到毫秒会导致 Seaborn 在计算 95% CI 时内存溢出并卡死。
    df['Time'] = df['Time'].round(0)

    # 排除极端录入错误 (如 ALT > 10000 的纯粹人工录入失误)，保证图表美观
    for bio in ['ALT (U/L)', 'AST (U/L)', 'TBIL (umol/L)']:
        mask = df['Biomarker'] == bio
        q99 = df[mask]['Value'].quantile(0.995)
        df.loc[mask, 'Value'] = df.loc[mask, 'Value'].clip(upper=q99)

    return df[['Time', 'Value', 'Label', 'Biomarker']]

def generate_divergence_plot(settings=None):
    settings = settings or load_settings()
    seed_everything(settings.reproducibility)
    fig_dir = str(settings.paths.figures)
    os.makedirs(fig_dir, exist_ok=True)
    
    print("⏳ Rendering REAL Longitudinal Biomarker Divergence Plot...")
    df = load_real_longitudinal_data(settings)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    biomarkers = ['ALT (U/L)', 'AST (U/L)', 'TBIL (umol/L)']
    titles = ['A. Alanine Aminotransferase (ALT)', 'B. Aspartate Aminotransferase (AST)', 'C. Total Bilirubin (TBIL)']
    
    palette = {'DILI Positive': COLOR_POS, 'Negative Control': COLOR_NEG}
    
    for idx, (ax, biomarker, title) in enumerate(zip(axes, biomarkers, titles)):
        sub_df = df[df['Biomarker'] == biomarker]
        
        # 绘制真实的置信区间时序图 (Bootstrapped 95% CI)
        # n_boot 调小一点 (如 500) 可以在保持科学严谨的同时提升渲染速度
        sns.lineplot(data=sub_df, x='Time', y='Value', hue='Label', 
                     palette=palette, linewidth=3, errorbar=('ci', 95), n_boot=500,
                     seed=settings.reproducibility.bootstrap_seed,
                     ax=ax, legend=(idx == 0))
        
        # 临床参考上限线 (Upper Limit of Normal, ULN)
        uln = 40 if 'ALT' in biomarker or 'AST' in biomarker else 21
        ax.axhline(y=uln, color='#757575', linestyle='--', linewidth=1.5, zorder=0)
        
        # 【修复1】动态计算 Y 轴跨度，强制向上扩展 10% 的绘图空间，确保文字永远有空间展示
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, max(y_max, uln) + (max(y_max, uln) - y_min) * 0.1)
        y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
        
        # 文字坐标改为动态百分比，紧贴虚线上方 2%
        # ax.text(-160, uln + y_span * 0.02, 'Upper Limit of Normal (ULN)', 
        #         color='#757575', fontsize=10, fontweight='bold')

        # 文字坐标下调，置于虚线下方 3% 处，并增加 va='top' 约束
        ax.text(-160, uln - y_span * 0.03, 'Upper Limit of Normal (ULN)', 
                color='#757575', fontsize=10, fontweight='bold', va='top')
        
        # 标记 72 小时黄金预警线
        ax.axvline(x=-72, color='#E5A9A9', linestyle=':', linewidth=2.5, zorder=0)
        if idx == 0:
            # 【修复2】将文字的 Y 坐标大幅上移至画布 80% 高度处，改用居中对齐，彻底防止挤压 X 轴
            text_y = ax.get_ylim()[0] + y_span * 0.8
            ax.text(-75, text_y, '72h Divergence Horizon', 
                    color='#8B0000', fontsize=11, fontweight='bold', ha='right', va='center', rotation=90)
        
        ax.set_title(title, loc='left', fontweight='bold', fontsize=16, pad=15)
        ax.set_xlabel('Time to DILI Onset Event (Hours)', fontweight='bold')
        ax.set_ylabel(biomarker, fontweight='bold')
        ax.set_xlim([-168, 0])
        
        ax.set_xticks([-168, -120, -72, -24, 0])
        ax.set_xticklabels(['-168h\n(Day 7)', '-120h\n(Day 5)', '-72h\n(Day 3)', '-24h\n(Day 1)', '0h\n(Onset)'], fontweight='bold')
        
        ax.grid(True, linestyle=':', alpha=0.6)
        sns.despine(ax=ax)
        
        if idx == 0:
            # 将 loc 从 'upper left' 修改为 'center left'
            ax.legend(loc='lower right', frameon=False, fontsize=11, title='Cohort Trajectory', title_fontproperties={'weight':'bold'})
            
    # -----------------------------------------------------------------
    # 导出
    # -----------------------------------------------------------------
    plt.tight_layout(pad=2.0)
    out_path_png = os.path.join(fig_dir, "Fig_1d_Biomarker_Divergence.png")
    out_path_pdf = os.path.join(fig_dir, "Fig_1d_Biomarker_Divergence.pdf")
    
    plt.savefig(out_path_png, dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig(out_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n🎉 完美！基于真实化验数据的 Figure 1d 纵向生化轨迹图渲染完毕！(400 DPI)")
    print(f"👉 查阅路径: {out_path_png}")

if __name__ == "__main__":
    generate_divergence_plot()

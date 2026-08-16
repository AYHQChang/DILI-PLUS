"""
DILI-PLUS | Figure 5：单病例药物归因（包实现）

职责：将 06b 输出的药物 Integrated Gradients 分数绘制为局部归因瀑布图。
输入：reports/06b_Target_Patient_Attribution.csv。
输出：figures/Fig_5_IG_Waterfall_Attribution.png 和 PDF。
状态：当前局部解释制图脚本；正负归因表示模型输出方向，不等同于肝毒性或保护作用。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

from diliplus.config import load_settings

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

COLOR_RISK_UP = '#DF9E9B'    
COLOR_RISK_DOWN = '#99CDCE'  

# =============================================================================
# 🚀 瀑布图渲染引擎
# =============================================================================
def generate_waterfall_chart(settings=None):
    settings = settings or load_settings()
    data_path = os.path.join(settings.paths.reports, "06b_Target_Patient_Attribution.csv")
    fig_dir = str(settings.paths.figures)
    os.makedirs(fig_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"🚨 Data not found: {data_path}")
        return
        
    print("⏳ Loading Integrated Gradients Attribution Data...")
    df = pd.read_csv(data_path)
    
    df_pos = df[df['Attribution_Score'] > 0].sort_values(by='Attribution_Score', ascending=False)
    df_neg = df[df['Attribution_Score'] < 0].sort_values(by='Attribution_Score', ascending=True)
    df_plot = pd.concat([df_pos, df_neg]).reset_index(drop=True)
    
    # 🔥 修复 1: 使用 textwrap 对极长英文药名进行智能断行 (宽度限制 15 字符)
    raw_labels = df_plot['Medication_EN'].tolist()
    labels = [textwrap.fill(lbl, width=15) for lbl in raw_labels]
    
    scores = df_plot['Attribution_Score'].tolist()
    pcts = df_plot['Contribution_Pct'].tolist()
    
    bottoms = []
    cumulative_score = 0.0
    for s in scores:
        bottoms.append(cumulative_score)
        cumulative_score += s

    # -----------------------------------------------------------------
    # 🎨 开始绘制高定瀑布图
    # -----------------------------------------------------------------
    print("🎨 Painting Figure 5: Microscopic Pathogenesis Waterfall...")
    fig, ax = plt.subplots(figsize=(15, 8))
    
    colors = [COLOR_RISK_UP if s > 0 else COLOR_RISK_DOWN for s in scores]
    bars = ax.bar(labels, scores, bottom=bottoms, color=colors, edgecolor='black', linewidth=1.5, width=0.55, zorder=3)
    
    for i in range(1, len(scores)):
        ax.plot([i-1, i], [bottoms[i], bottoms[i]], color='gray', linestyle='--', linewidth=1.5, zorder=2)
        
    for i, (bar, score, pct) in enumerate(zip(bars, scores, pcts)):
        if score > 0:
            y_text = bottoms[i] + score + 0.003
            va = 'bottom'
            sign = '+'
        else:
            y_text = bottoms[i] + score - 0.003
            va = 'top'
            sign = ''
            
        ax.text(bar.get_x() + bar.get_width() / 2, y_text, 
                f"{sign}{score:.3f}\n({sign}{pct:.1f}%)", 
                ha='center', va=va, fontweight='bold', fontsize=11, color='#333333')

    # 🔥 修复 2: 动态计算全局最极端的 Y 轴值，并强制预留 25% 的上下缓冲带
    all_y_points = bottoms + [bottoms[i] + scores[i] for i in range(len(scores))]
    y_min, y_max = min(all_y_points), max(all_y_points)
    y_range = y_max - y_min if (y_max - y_min) != 0 else 0.1
    ax.set_ylim(y_min - y_range * 0.25, y_max + y_range * 0.25)

    patient_id = df['Patient_ID'].iloc[0]
    ax.set_title(f"Integrated Gradients Attribution for Patient {patient_id}", loc='left', fontweight='bold', fontsize=18, pad=20)
    ax.set_ylabel(r"$\Delta$ Prediction Risk Score (vs. Baseline)", fontweight='bold', fontsize=14)
    ax.set_xlabel("Administered Medications (Temporal Sequence Overlaid)", fontweight='bold', fontsize=14, labelpad=15)
    
    # 将 X 轴标签角度调整为 30 度，配合换行使其更加优雅
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontweight='bold', fontsize=12)
    
    ax.axhline(0, color='black', linewidth=2.0, zorder=1)
    ax.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)
    sns.despine(ax=ax)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_RISK_UP, edgecolor='black', label=r'Hepatotoxic Contribution (Risk $\uparrow$)'),
        Patch(facecolor=COLOR_RISK_DOWN, edgecolor='black', label=r'Hepatoprotective Contribution (Risk $\downarrow$)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=12)

    plt.tight_layout(pad=3.0)
    out_path_png = os.path.join(fig_dir, "Fig_5_IG_Waterfall_Attribution.png")
    out_path_pdf = os.path.join(fig_dir, "Fig_5_IG_Waterfall_Attribution.pdf")
    
    plt.savefig(out_path_png, dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig(out_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n🎉 完美！优化边界后的 Figure 5 渲染完毕！(400 DPI)")
    print(f"👉 查阅路径: {out_path_png}")

if __name__ == "__main__":
    generate_waterfall_chart()

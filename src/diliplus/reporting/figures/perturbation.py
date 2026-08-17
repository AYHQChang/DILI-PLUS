"""
DILI-PLUS | Figure 6 当前版：药物扰动与替换敏感性（包实现）

职责：将 06c 输出绘制为可扩展的药物嵌入缩放轨迹和 Token 替换前后概率比较。
输入：reports/06c_Counterfactual_Trajectory.csv。
输出：figures/Fig_6_InSilico_Simulation.png 和 PDF。
状态：当前 Figure 6 制图脚本；文件名中的“11”是历史版本痕迹。
解释边界：图中结果是模型预测敏感性，不是剂量反应、药效或临床干预效应。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

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

# 扩展版学术高级色卡 (支持十种以上药物的高对比度隔离)
EXTENDED_PALETTE = [
    '#DF9E9B', '#99CDCE', '#F8BF92', '#99BADF', '#999ACD', 
    '#FFB3DD', '#A8E6CF', '#FFD3B6', '#D4A5A5', '#9DC8C8', 
    '#B5B8D3', '#F4B6C2', '#CDE5DCE', '#E8D5C4', '#A2D5AB'
]

# 扩展版高辨识度几何标记 (排除易混淆形状)
EXTENDED_MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '<']

# Panel B 的输入状态配色
COLOR_ORIGINAL = '#999ACD'
COLOR_SUBSTITUTED = '#99CDCE'

def generate_simulation_figure(settings=None):
    settings = settings or load_settings()
    data_path = os.path.join(settings.paths.reports, "06c_Counterfactual_Trajectory.csv")
    fig_dir = str(settings.paths.figures)
    os.makedirs(fig_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"🚨 Data not found: {data_path}")
        return
        
    print("⏳ Loading medication-token perturbation data...")
    df = pd.read_csv(data_path)
    row_index = df['Patient_ID'].iloc[0]
    
    # -----------------------------------------------------------------
    # 🎨 准备 1x2 高级排版画布
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.8, 1]})
    ax_taper = axes[0]
    ax_subst = axes[1]
    
    # ==========================================
    # Panel A: medication-embedding attenuation
    # ==========================================
    df_taper = df[df['Intervention_Type'] == 'Dose Tapering'].copy()
    
    # 提取 Alpha 数值作为 X 轴
    df_taper['Alpha_Val'] = df_taper['Parameter'].apply(lambda x: float(x.split('=')[1]))
    medications = df_taper['Targeted_Medications_EN'].unique()
    
    # 🔥 初始化无限循环生成器，确保极端冗余下的色彩/形状分配不越界
    color_cycle = itertools.cycle(EXTENDED_PALETTE)
    marker_cycle = itertools.cycle(EXTENDED_MARKERS)
    
    for med in medications:
        med_df = df_taper[df_taper['Targeted_Medications_EN'] == med].sort_values('Alpha_Val', ascending=False)
        if len(med_df) == 0: continue
            
        risk_vals = med_df['Predicted_DILI_Risk'].values
        clean_label = med
        for tag in ('[*]', '[Max Variance]', '[Clinical Target]', '[Clinical Swap]'):
            clean_label = clean_label.replace(tag, '')
        clean_label = clean_label.strip()
        
        # 动态分配唯一标识对
        current_color = next(color_cycle)
        current_marker = next(marker_cycle)
        
        ax_taper.plot(med_df['Alpha_Val'], risk_vals, marker=current_marker, color=current_color, 
                      lw=3.0, markersize=8, alpha=0.9, label=clean_label, markeredgecolor='white', zorder=4)

    ax_taper.set_xlim([1.05, -0.05]) 
    ax_taper.set_title(
        f'A. Medication-Embedding Attenuation (Row {row_index})',
        loc='left', fontweight='bold', fontsize=17, pad=15
    )
    ax_taper.set_xlabel(r'Embedding amplitude multiplier ($\alpha$)', fontweight='bold', fontsize=14)
    ax_taper.set_ylabel('Model-predicted AHI-proxy probability (%)', fontweight='bold', fontsize=14)
    
    ax_taper.set_xticks([1.0, 0.75, 0.5, 0.25, 0.0])
    ax_taper.set_xticklabels(['1.0\n(Original)', '0.75', '0.50', '0.25', '0.0\n(Zero vector)'], fontweight='bold')
    
    ax_taper.grid(True, linestyle='--', alpha=0.6, zorder=0)
    sns.despine(ax=ax_taper)
    
    # 将图例放在左上角并使用无边框样式
    handles, labels = ax_taper.get_legend_handles_labels()
    ax_taper.legend(handles, labels, loc='upper left', 
                    frameon=False, fontsize=11, title='Perturbed medication token', title_fontproperties={'weight':'bold'})

    # ==========================================
    # Panel B: predefined medication-token substitution
    # ==========================================
    df_subst = df[df['Intervention_Type'] == 'Substitution']
    
    if not df_subst.empty:
        baseline_risk = df_taper[(df_taper['Targeted_Medications_EN'].str.contains('Atorvastatin')) & 
                                 (df_taper['Alpha_Val'] == 1.0)]['Predicted_DILI_Risk'].values[0]
        
        subst_row = df_subst.iloc[0]
        subst_risk = subst_row['Predicted_DILI_Risk']
        delta_model_pp = subst_risk - baseline_risk
        
        scenarios = ['Original input token\n(Atorvastatin)', 'Substituted input token\n(Pravastatin)']
        risks = [baseline_risk, subst_risk]
        bar_colors = [COLOR_ORIGINAL, COLOR_SUBSTITUTED]
        
        bars = ax_subst.bar(scenarios, risks, color=bar_colors, edgecolor='black', width=0.5, linewidth=2, zorder=3)
        
        for bar, risk in zip(bars, risks):
            ax_subst.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, 
                          f"{risk:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=14)
                          
        ax_subst.annotate(
            rf"$\Delta p_{{model}}$ = {delta_model_pp:+.1f} pp",
            # 修正：xytext保持0.5不动，仅修改箭头目标坐标xy为0.75（即1.0向左平移半个柱宽0.25）
            xy=(0.75, subst_risk), xytext=(0.5, max(risks) * 1.16),
            arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.8),
            ha='center', va='center', fontweight='bold', fontsize=12, color='#7A1F1F'
        )
                          
        ax_subst.set_ylim([0, max(risks) * 1.32])
        ax_subst.set_title('B. Predefined Token-Substitution Sensitivity', loc='left', fontweight='bold', fontsize=17, pad=15)
        ax_subst.set_xlabel('Medication-token input', fontweight='bold', fontsize=14)
        ax_subst.set_ylabel('Model-predicted AHI-proxy probability (%)', fontweight='bold', fontsize=14)
        ax_subst.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
        sns.despine(ax=ax_subst)
    
    # -----------------------------------------------------------------
    # 导出 PNG 与 PDF
    # -----------------------------------------------------------------
    plt.tight_layout(pad=3.0)
    out_path_png = os.path.join(fig_dir, "Fig_6_InSilico_Simulation.png")
    out_path_pdf = os.path.join(fig_dir, "Fig_6_InSilico_Simulation.pdf")
    
    plt.savefig(out_path_png, dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig(out_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Figure 6 medication-token sensitivity plot rendered (400 DPI).")
    print(f"👉 查阅路径: {out_path_png}")

if __name__ == "__main__":
    generate_simulation_figure()
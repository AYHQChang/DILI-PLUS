"""
DILI-PLUS | Figure 6 备选版：多面板药物扰动轨迹（已归档）

职责：将药物嵌入缩放与 Token 替换结果绘制为 2×3 单药轨迹面板。
输入：reports/06c_Counterfactual_Trajectory.csv。
输出：reports/figures/ 下的 Fig6_Counterfactual_Trajectories_DILI 文件。
状态：备选/历史版式；当前根目录 figures 中使用的是可扩展 1×2 版本。
解释边界：图中变化是模型敏感性，不是剂量反应、药效或临床换药效果。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# 与其他论文图保持一致的色卡
COLORS = {
    'tabert_mauve': '#A4799E',        # 风险增强/加剧 (替代原来的红色)
    'morandi_teal': '#5EA69C',        # 风险抑制/减轻 (替代原来的蓝色)
    'bert_pale_green': '#C2CFA2',     # 基线风险线 (替代原来的黄色)
    'lstm_dusty_purple': '#706690',   # Swap 替换框专用色 (高级科技感)
    'text_black': '#1A1A1A',
    'piano_stripe': '#F4F4F4',        # 图表背景色
    'deep_gray': '#555555'            # 坐标网格线
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Microsoft YaHei', 'SimHei'],
    'axes.unicode_minus': False,  
    'font.size': 12,
    'axes.linewidth': 1.5,
    'axes.edgecolor': COLORS['text_black'],
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'text.color': COLORS['text_black'],
    'axes.labelcolor': COLORS['text_black'],
    'xtick.color': COLORS['text_black'],
    'ytick.color': COLORS['text_black']
})

def extract_alpha(param_str):
    try:
        if 'Alpha=' in str(param_str):
            return float(param_str.replace('Alpha=', ''))
    except Exception:
        pass
    return None

def plot_counterfactual_trajectories():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(base_dir, "reports")
    output_dir = os.path.join(reports_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # 动态匹配路径
    csv_path = os.path.join(reports_dir, "06c_Counterfactual_Trajectory.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "06c_Counterfactual_Trajectory.csv")
        
    if not os.path.exists(csv_path):
        print(f"🚨 Missing Counterfactual Data. Please run 06c first. Looked for {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    global_base_series = df[df['Parameter'] == 'Alpha=1.0']['Predicted_DILI_Risk']
    global_base_risk = global_base_series.iloc[0] if not global_base_series.empty else 0.0
    
    enhancers = df[df['Drug_Role'].str.contains('Enhancer', case=False, na=False)].copy()
    suppressors = df[df['Drug_Role'].str.contains('Suppressor', case=False, na=False)].copy()
    
    enhancer_drugs = [d for d in enhancers['Targeted_Medications'].unique() if pd.notna(d)][:3]
    suppressor_drugs = [d for d in suppressors['Targeted_Medications'].unique() if pd.notna(d)][:3]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor('#FFFFFF') # 论文外背景纯白
    
    def plot_trajectory(row_idx, col_idx, drug_group, drug_zh_name, role):
        ax = axes[row_idx, col_idx]
        
        if drug_group.empty:
            ax.set_visible(False)
            return
            
        ax.set_facecolor('#FFFFFF')
        
        taper_group = drug_group[drug_group['Intervention_Type'] == 'Dose Tapering'].copy()
        if taper_group.empty:
            ax.set_visible(False)
            return
            
        taper_group['Alpha'] = taper_group['Parameter'].apply(extract_alpha)
        taper_group = taper_group.sort_values('Alpha', ascending=False)
        
        drug_en = taper_group['Targeted_Medications_EN'].iloc[0]
        
        x_vals = taper_group['Alpha'].values
        y_vals = taper_group['Predicted_DILI_Risk'].values
        
        base_risk_series = taper_group[taper_group['Alpha'] == 1.0]['Predicted_DILI_Risk']
        base_risk = base_risk_series.iloc[0] if not base_risk_series.empty else y_vals[0]
            
        # 优化点 1: 基线使用浅绿虚线
        ax.axhline(base_risk, color=COLORS['bert_pale_green'], linestyle='--', linewidth=2.5, zorder=2)
        
        is_enhancer = 'Enhancer' in role
        # 优化点 2: 颜色映射重构 (Mauve for Enhancer, Teal for Suppressor)
        line_color = COLORS['tabert_mauve'] if is_enhancer else COLORS['morandi_teal']
        marker_style = 'o' if is_enhancer else 's'
        
        ax.plot(x_vals, y_vals, color=line_color, marker=marker_style, markersize=8, 
                linewidth=3, zorder=4)
        
        ax.set_xlabel('Feature Presence Amplitude (Alpha)', fontweight='bold')
        ax.set_ylabel('Predicted DILI Risk (%)', fontweight='bold')
        ax.set_title(drug_en, fontweight='bold', pad=12)
        
        # 优化点 3: 阴影区域也采用同色系，避免红绿灯冲突
        ax.fill_between(x_vals, base_risk, y_vals, where=(y_vals > base_risk), 
                        interpolate=True, color=COLORS['tabert_mauve'], alpha=0.15, zorder=1)
        ax.fill_between(x_vals, base_risk, y_vals, where=(y_vals <= base_risk), 
                        interpolate=True, color=COLORS['morandi_teal'], alpha=0.20, zorder=1)
        
        ax.set_xticks(x_vals)
        ax.invert_xaxis() 
        ax.grid(True, linestyle='--', color=COLORS['deep_gray'], alpha=0.3, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # =====================================================================
        # 峰值标注与 Swap 智能锚定逻辑 (核心逻辑保留，仅改色)
        # =====================================================================
        abs_diffs = np.abs(y_vals - base_risk)
        max_idx = np.argmax(abs_diffs)
        peak_x = x_vals[max_idx]
        peak_y = y_vals[max_idx]
        diff_val = peak_y - base_risk
        
        sign_str = "+" if diff_val > 0 else ""
        
        fixed_ha = 'right'  
        fixed_va = 'center' 
        x_data_offset = 0.18 # 箭头留空距离
        
        # 1. 绘制 Max Δ 标注
        ax.annotate(f"Max Δ: {sign_str}{diff_val:.2f}%",
                    xy=(peak_x, peak_y),
                    xytext=(peak_x + x_data_offset, peak_y),
                    color=line_color, fontweight='bold', fontsize=11,
                    ha=fixed_ha, va=fixed_va,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=line_color, lw=1.5, alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color=line_color, lw=1.5, connectionstyle="arc3,rad=0.0"),
                    zorder=5)
        
        # 2. 🌟 提取 Swap Box 逻辑，紧贴 Max 标志上下方
        sub_group = drug_group[drug_group['Intervention_Type'] == 'Substitution']
        all_y_bounds = list(y_vals) + [base_risk]

        if not sub_group.empty:
            sub_risk = sub_group['Predicted_DILI_Risk'].iloc[0]
            sub_name_en = str(sub_group['Parameter'].iloc[0]).replace('Swap to ', '').strip()
            
            # 计算 Swap 带来的差值与符号
            sub_diff = sub_risk - base_risk
            sub_sign = "+" if sub_diff > 0 else ""

            y_min_temp, y_max_temp = min(all_y_bounds), max(all_y_bounds)
            dynamic_margin = max((y_max_temp - y_min_temp) * 0.25, 1.0)

            if row_idx == 0:
                swap_y = peak_y - (dynamic_margin * 0.45)
                swap_va = 'top'
            else:
                swap_y = peak_y + (dynamic_margin * 0.45)
                swap_va = 'bottom'

            swap_text = f"If swap to: {sub_name_en}\nRisk: {sub_risk:.2f}% ({sub_sign}{sub_diff:.2f}%)"
            
            # 优化点 4: 交换框采用高级紫灰 (lstm_dusty_purple)
            ax.text(peak_x + x_data_offset, swap_y, swap_text,
                    color=COLORS['lstm_dusty_purple'], fontweight='bold', fontsize=10,
                    ha=fixed_ha, va=swap_va,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS['lstm_dusty_purple'], lw=1.5, alpha=0.9),
                    zorder=7)

            all_y_bounds.extend([swap_y, sub_risk])

        # 边界保护
        y_min, y_max = min(all_y_bounds), max(all_y_bounds)
        margin = max((y_max - y_min) * 0.25, 1.0)
        ax.set_ylim(max(0, y_min - margin), y_max + margin)

    # 循环渲染 Enhancers
    for i in range(3):
        if i < len(enhancer_drugs):
            drug = enhancer_drugs[i]
            group = enhancers[enhancers['Targeted_Medications'] == drug].copy()
            plot_trajectory(0, i, group, drug, 'Enhancer')
        else:
            axes[0, i].set_visible(False)
            
    # 循环渲染 Suppressors
    for i in range(3):
        if i < len(suppressor_drugs):
            drug = suppressor_drugs[i]
            group = suppressors[suppressors['Targeted_Medications'] == drug].copy()
            plot_trajectory(1, i, group, drug, 'Suppressor')
        else:
            axes[1, i].set_visible(False)
            
    # 全局混合图例
    legend_elements = [
        Line2D([0], [0], color=COLORS['tabert_mauve'], lw=3, marker='o', markersize=8, label='Risk Enhancer (Pathogenic) Tapering'),
        Line2D([0], [0], color=COLORS['morandi_teal'], lw=3, marker='s', markersize=8, label='Risk Suppressor (Protective) Tapering'),
        Line2D([0], [0], color=COLORS['bert_pale_green'], lw=2.5, linestyle='--', label=f'Baseline DILI Risk (Alpha=1.0, {global_base_risk:.2f}%)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markeredgecolor=COLORS['lstm_dusty_purple'], markeredgewidth=1.5, markersize=12, label='Counterfactual Substitution Panel'),
        Patch(facecolor=COLORS['morandi_teal'], alpha=0.20, label='Risk Reduction Zone (Favourable)'),
        Patch(facecolor=COLORS['tabert_mauve'], alpha=0.15, label='Risk Escalation Zone (Adverse)')
    ]
    
    # 移除图例的强外框，与全文风格保持一致
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, frameon=False, fontsize=12, title="Pharmacodynamic Counterfactual Intervention Strategy", title_fontsize=14)
               
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.35, wspace=0.25)
    
    final_pdf = os.path.join(output_dir, "Fig6_Counterfactual_Trajectories_DILI.pdf")
    final_png = os.path.join(output_dir, "Fig6_Counterfactual_Trajectories_DILI.png")
    
    plt.savefig(final_pdf)
    plt.savefig(final_png)
    print(f"\n✅ [Plotting Complete] High-res figures saved to:\n   -> {final_pdf}\n   -> {final_png}")

if __name__ == "__main__":
    plot_counterfactual_trajectories()

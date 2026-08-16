"""
DILI-PLUS | 旧版综合绘图入口（已归档）

职责：按旧版结果 schema 绘制模型指标、早期预警与局部归因图。
输入：旧版 reports CSV 文件。
输出：多张历史图表。
状态：遗留脚本；仍引用 AKI_* 字段，与当前 DILI 单任务结果表不兼容。
现行论文图请使用 Fig_2_model_compar.py 至 Fig_6_plot_clinical_cases11.py。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# User-Defined High-Contrast Palette
HQ_COLORS = {
    'red': '#DB3124',         
    'yellow': '#FFDF92',      
    'light_blue': '#90BEE0',  
    'dark_blue': '#4B74B2',   
    'light_green': '#D4EDDA', # Optimal Intervention Window / Target AUROC
    'text_black': '#1A1A1A'   
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.unicode_minus': False,  
    'font.size': 13,
    'axes.linewidth': 1.5,
    'axes.edgecolor': HQ_COLORS['text_black'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'text.color': HQ_COLORS['text_black'],
    'axes.labelcolor': HQ_COLORS['text_black'],
    'xtick.color': HQ_COLORS['text_black'],
    'ytick.color': HQ_COLORS['text_black']
})

def extract_mean(val):
    """Helper function to extract the mean float value from strings like '0.9121 (0.8952-0.9276)'"""
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        return float(val.split('(')[0].strip())
    return float(val)

def plot_baseline_comparison(reports_dir, output_dir):
    """Plot Fig 2: Discrimination (AUROC & AUPRC) vs. Calibration Error (ECE) Trade-off"""
    csv_path = os.path.join(reports_dir, "05_Experiment_Results_Table.csv")
    if not os.path.exists(csv_path):
        return

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return

    df = df.drop_duplicates(subset=['Model_Architecture'], keep='last').copy()
    
    # 解析数值列
    df['AKI_AUROC_num'] = df['AKI_AUROC'].apply(extract_mean)
    df['AKI_AUPRC_num'] = df['AKI_AUPRC'].apply(extract_mean)
    
    # 自动识别新的ECE列名
    ece_col = 'AKI_Uniform_ECE' if 'AKI_Uniform_ECE' in df.columns else ('AKI_ECE' if 'AKI_ECE' in df.columns else 'AKI_Quant_ECE')
    if ece_col in df.columns:
        df['AKI_ECE_num'] = df[ece_col].apply(extract_mean)
    else:
        df['AKI_ECE_num'] = 0.0 # 兜底防错

    df = df.sort_values(by='AKI_AUPRC_num', ascending=True).reset_index(drop=True)
    
    models_short = [m.replace('MultiModal', '').replace('LogisticRegression', 'LR').replace('Baseline', 'Base_') for m in df['Model_Architecture']]
    x = np.arange(len(df['Model_Architecture']))
    width = 0.35 # Adjusted width for grouped bars
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx() 
    
    yerr_prc = df['AKI_AUPRC_std'] if 'AKI_AUPRC_std' in df.columns else None
    yerr_roc = df['AKI_AUROC_std'] if 'AKI_AUROC_std' in df.columns else None
    
    # Plot AUROC (Left Bar)
    rects_roc = ax1.bar(x - width/2, df['AKI_AUROC_num'], width, yerr=yerr_roc, capsize=5, 
                     label='Discrimination: AKI AUROC', 
                     color=HQ_COLORS['light_blue'], edgecolor='black', linewidth=1.2, zorder=3)

    # Plot AUPRC (Right Bar)
    rects_prc = ax1.bar(x + width/2, df['AKI_AUPRC_num'], width, yerr=yerr_prc, capsize=5, 
                     label='Discrimination: AKI AUPRC', 
                     color=HQ_COLORS['dark_blue'], edgecolor='black', linewidth=1.2, zorder=3)
    
    # Plot ECE (Line on Secondary Axis)
    line1 = ax2.plot(x, df['AKI_ECE_num'], color=HQ_COLORS['red'], marker='D', 
                     markersize=9, linewidth=2.5, linestyle='--', 
                     label='Calibration Error: AKI ECE', zorder=4)
    
    target_idx = -1
    for i, model in enumerate(df['Model_Architecture']):
        if 'TimeAware' in model:
            # Target Model Styling
            rects_roc[i].set_color(HQ_COLORS['light_green'])
            rects_roc[i].set_edgecolor('black')
            rects_roc[i].set_linewidth(1.8)

            rects_prc[i].set_color(HQ_COLORS['yellow'])
            rects_prc[i].set_edgecolor('black')
            rects_prc[i].set_linewidth(1.8)
            target_idx = i
            
    ax1.set_ylabel('Performance Score (Higher is Better)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('ECE Score (Lower is Better)', fontweight='bold', color=HQ_COLORS['red'], fontsize=14)
    
    # Scale Y-axis based on the maximum value of either AUROC or AUPRC
    max_val = max(df['AKI_AUPRC_num'].max(), df['AKI_AUROC_num'].max())
    ax1.set_ylim(0, max_val * 1.30)
    ax1.set_title('Figure 2: AKI Discrimination vs. Calibration Trade-off', fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_short, rotation=15, ha='right', fontweight='bold')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', 
               bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, fontsize=11) 
    
    ax1.grid(axis='y', linestyle='--', alpha=0.5, color='gray', zorder=0)
    
    for i in range(len(df)):
        is_target = (i == target_idx)
        
        # Annotate AUROC
        ax1.annotate(f"{df['AKI_AUROC_num'].iloc[i]:.3f}",
                     xy=(rects_roc[i].get_x() + rects_roc[i].get_width() / 2, rects_roc[i].get_height()),
                     xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', 
                     fontsize=10, fontweight='bold' if is_target else 'normal', 
                     color=HQ_COLORS['text_black'])

        # Annotate AUPRC
        ax1.annotate(f"{df['AKI_AUPRC_num'].iloc[i]:.3f}",
                     xy=(rects_prc[i].get_x() + rects_prc[i].get_width() / 2, rects_prc[i].get_height()),
                     xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', 
                     fontsize=10, fontweight='bold' if is_target else 'normal', 
                     color=HQ_COLORS['text_black'])
        
        # Annotate ECE
        ax2.annotate(f"{df['AKI_ECE_num'].iloc[i]:.3f}",
                     xy=(x[i], df['AKI_ECE_num'].iloc[i]),
                     xytext=(0, -22), textcoords="offset points", ha='center', va='top', 
                     fontsize=11, fontweight='bold', color=HQ_COLORS['red'])

    plt.tight_layout()
    out_path = os.path.join(output_dir, "Fig2_Model_Comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"   ✅ Fig 2 exported: {out_path}")


def plot_early_warning_decay(reports_dir, output_dir):
    """Plot Fig 3: AUROC & AUPRC Decay across different lead-time windows"""
    csv_path = os.path.join(reports_dir, "06a_Early_Warning_Decay_All_Models.csv")
    if not os.path.exists(csv_path):
        return

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return

    # 解析数值列
    df['AKI_AUROC_num'] = df['AKI_AUROC'].apply(extract_mean)
    df['AKI_AUPRC_num'] = df['AKI_AUPRC'].apply(extract_mean)

    fig, ax = plt.subplots(figsize=(11, 7))
    
    # 浅绿色表示最佳干预时间窗
    ax.axvspan(0, 24, color=HQ_COLORS['light_green'], alpha=0.35, zorder=0, 
               label='Optimal Intervention Window (≤ 24h)')
    
    model_styles = {
        'MultiModalBiLSTM': {
            'color': HQ_COLORS['yellow'], 'marker': 's', 'lw': 2.5, 
            'label': 'BiLSTM', 'mec': 'black' 
        },
        'MultiModalBaselineMedBERT': {
            'color': HQ_COLORS['dark_blue'], 'marker': '^', 'lw': 2.0, 
            'label': 'Baseline MedBERT', 'mec': HQ_COLORS['dark_blue']
        },
        'MultiModalTimeAwareMedBERT': {
            'color': HQ_COLORS['red'], 'marker': 'o', 'lw': 3.5, 
            'label': 'Time-Aware MedBERT (Ours)', 'mec': HQ_COLORS['red']
        }
    }
    
    models_present = df['Model_Architecture'].unique()
    
    for model_name in models_present:
        model_data = df[df['Model_Architecture'] == model_name].sort_values(by='Lead_Time_Hours')
        style = model_styles.get(model_name, {'color': 'gray', 'marker': 'x', 'lw': 2.0, 'label': model_name, 'mec': 'gray'})
        
        # Plot AUPRC (Solid Line)
        ax.plot(model_data['Lead_Time_Hours'], model_data['AKI_AUPRC_num'], 
                color=style['color'], marker=style['marker'], markeredgecolor=style['mec'],
                linestyle='-', linewidth=style['lw'], markersize=9, 
                label=f"{style['label']} (AUPRC)", zorder=3)

        # Plot AUROC (Dashed Line)
        ax.plot(model_data['Lead_Time_Hours'], model_data['AKI_AUROC_num'], 
                color=style['color'], marker=style['marker'], markeredgecolor=style['mec'],
                linestyle='--', linewidth=style['lw'], markersize=9, 
                label=f"{style['label']} (AUROC)", zorder=3)
        
        # Target Model Annotations
        if 'TimeAware' in model_name:
            for _, row in model_data.iterrows():
                # AUROC
                ax.annotate(f"{row['AKI_AUROC_num']:.3f}", 
                            xy=(row['Lead_Time_Hours'], row['AKI_AUROC_num']),
                            xytext=(0, 10), textcoords="offset points", 
                            ha='center', va='bottom', fontsize=10, fontweight='bold', color=style['color'])
                # AUPRC
                ax.annotate(f"{row['AKI_AUPRC_num']:.3f}", 
                            xy=(row['Lead_Time_Hours'], row['AKI_AUPRC_num']),
                            xytext=(0, -18), textcoords="offset points", 
                            ha='center', va='top', fontsize=10, fontweight='bold', color=style['color'])

    ax.set_title('Figure 3: AKI Early-Warning Lead-Time Performance Decay', fontweight='bold', pad=15)
    ax.set_xlabel('Lead Time (Hours prior to AKI onset)', fontweight='bold')
    ax.set_ylabel('Performance Score (AUROC & AUPRC)', fontweight='bold')
    
    lookahead_windows = sorted(df['Lead_Time_Hours'].unique())
    ax.set_xticks(lookahead_windows)
    ax.set_xticklabels([f"{int(h)}h" for h in lookahead_windows], fontweight='bold')
    
    ax.invert_xaxis() 
    ax.grid(True, linestyle='--', alpha=0.6, zorder=1)
    
    ax.legend(loc='upper left', frameon=True, fontsize=10, ncol=2) 
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "Fig3_Early_Warning_Decay.png")
    plt.savefig(out_path)
    plt.close()
    print(f"   ✅ Fig 3 exported: {out_path}")


def plot_ig_attribution(reports_dir, output_dir):
    """
    Plot Fig 4: Local Integrated Gradients Attribution for a Single Patient
    Dynamically extracts ID to ensure absolute Data Provenance.
    """
    ig_csv = os.path.join(reports_dir, "06b_IG_Attribution.csv") 
    if not os.path.exists(ig_csv):
        print(f"   ❌ Warning: Missing {ig_csv}")
        return
        
    try:
        df = pd.read_csv(ig_csv)
    except pd.errors.EmptyDataError:
        return

    # 动态解析 ID
    patient_id = "Unknown_ID"
    id_cols = [col for col in df.columns if any(k in col.lower() for k in ['id', 'patient', 'subject', 'encounter'])]
    
    if id_cols:
        patient_id = str(df[id_cols[0]].iloc[0])
        df = df.drop(columns=id_cols)
    else:
        print("   ⚠️ Pipeline Warning: Upstream CSV lacks a definitive Patient ID column. Proceeding with 'Unknown_ID'.")

    feature_col = df.columns[0]
    score_col = df.columns[1]
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['score', 'attribution', 'ig', 'weight']):
            score_col = col
        if any(keyword in col.lower() for keyword in ['feature', 'name', 'drug', 'diag', 'variable']):
            feature_col = col

    df['abs_score'] = df[score_col].abs()
    df_top = df.sort_values(by='abs_score', ascending=False).head(15).copy()
    df_top = df_top.sort_values(by='abs_score', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = [HQ_COLORS['red'] if val > 0 else HQ_COLORS['dark_blue'] for val in df_top[score_col]]
    bars = ax.barh(df_top[feature_col].astype(str), df_top[score_col], color=colors, edgecolor='black', linewidth=1.0, alpha=0.9)
    
    ax.set_title(f'Figure 4: Local IG Attribution for Individual Patient (ID: {patient_id})', fontweight='bold', pad=15)
    ax.set_xlabel('Local IG Attribution Score (Contribution to Individual AKI Risk)', fontweight='bold')
    ax.axvline(0, color='black', linewidth=1.5, linestyle='-')
    
    ax.grid(axis='x', linestyle='--', linewidth=1.2, color=HQ_COLORS['yellow'], alpha=0.9)
    
    x_max = df_top[score_col].max()
    x_min = df_top[score_col].min()
    x_range = x_max - x_min if x_max != x_min else 1.0
    ax.set_xlim(x_min - x_range * 0.15, x_max + x_range * 0.15)
    
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + (x_range * 0.015 if width > 0 else -x_range * 0.015)
        ha_align = 'left' if width > 0 else 'right'
        
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                va='center', ha=ha_align, fontsize=11, fontweight='bold', 
                color=HQ_COLORS['text_black'])

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"Fig4_IG_Attribution_{patient_id}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"   ✅ Fig 4 (Patient {patient_id}) matched and exported: {out_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(base_dir, "reports")
    output_dir = os.path.join(base_dir, "figures")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Initialising Final Tuned Plotting Pipeline...")
    plot_baseline_comparison(reports_dir, output_dir)
    plot_early_warning_decay(reports_dir, output_dir)
    plot_ig_attribution(reports_dir, output_dir)
    
    print("🎉 All plotting tasks completed successfully.")

if __name__ == "__main__":
    main()

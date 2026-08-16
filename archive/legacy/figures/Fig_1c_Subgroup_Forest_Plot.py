"""
DILI-PLUS | Figure 1c：亚组森林图版式原型（已归档）

职责：用内置示例数据演示亚组 AUROC、95% CI 和组间 P 值的森林图布局。
输入：generate_mock_subgroup_data() 中硬编码的模拟数值。
输出：figures/Fig_1c_Subgroup_Forest_Plot.png 和 PDF。
状态：占位制图脚本，不包含真实亚组评估，不能作为研究结果或公平性证据。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 🎨 顶刊级审美设定
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 12,
    'axes.linewidth': 1.5,
    'figure.dpi': 300,
    'text.color': '#222222'
})

COLOR_DOT = '#DF9E9B'      # 核心模型色 (粉红)
COLOR_LINE = '#333333'     # 误差线 (深灰)
COLOR_REF = '#99CDCE'      # 整体均值参考线 (青色)

def generate_mock_subgroup_data():
    """
    生成高度逼真的亚组评估数据。
    在实际流程中，您可以通过 06a 脚本切分 DataLoader 并计算各组真实的 AUROC 替换此数据。
    """
    data = [
        {'Category': 'Overall', 'Subgroup': 'All Patients', 'N': 51316, 'AUROC': 0.885, 'CI_L': 0.871, 'CI_U': 0.898, 'P_val': None},
        
        {'Category': 'Age', 'Subgroup': '< 65 years', 'N': 32104, 'AUROC': 0.891, 'CI_L': 0.875, 'CI_U': 0.905, 'P_val': 'Ref'},
        {'Category': 'Age', 'Subgroup': '>= 65 years', 'N': 19212, 'AUROC': 0.874, 'CI_L': 0.852, 'CI_U': 0.895, 'P_val': '0.12'},
        
        {'Category': 'Sex', 'Subgroup': 'Male', 'N': 28451, 'AUROC': 0.882, 'CI_L': 0.865, 'CI_U': 0.899, 'P_val': 'Ref'},
        {'Category': 'Sex', 'Subgroup': 'Female', 'N': 22865, 'AUROC': 0.889, 'CI_L': 0.870, 'CI_U': 0.906, 'P_val': '0.34'},
        
        {'Category': 'Polypharmacy', 'Subgroup': '<= 3 concurrent drugs', 'N': 15420, 'AUROC': 0.912, 'CI_L': 0.895, 'CI_U': 0.928, 'P_val': 'Ref'},
        {'Category': 'Polypharmacy', 'Subgroup': '> 3 concurrent drugs', 'N': 35896, 'AUROC': 0.865, 'CI_L': 0.845, 'CI_U': 0.882, 'P_val': '<0.01'},
        
        {'Category': 'Baseline Liver Disease', 'Subgroup': 'No', 'N': 42100, 'AUROC': 0.895, 'CI_L': 0.880, 'CI_U': 0.910, 'P_val': 'Ref'},
        {'Category': 'Baseline Liver Disease', 'Subgroup': 'Yes', 'N': 9216, 'AUROC': 0.842, 'CI_L': 0.815, 'CI_U': 0.868, 'P_val': '<0.01'}
    ]
    return pd.DataFrame(data)

def generate_forest_plot():
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    print("⏳ Rendering High-Resolution Forest Plot...")
    df = generate_mock_subgroup_data()
    
    # 反转 DataFrame 顺序，因为图表坐标是从下往上绘制的
    df = df.iloc[::-1].reset_index(drop=True)
    
    # 画布设定：回归严谨的 10:6 学术宽银幕比例，避免垂直留白过剩
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 核心绘图区数据锚点（保持不变，作为后续文本坐标系的基准参照）
    x_min, x_max = 0.75, 0.95
    
    # 【逻辑修复核心】彻底解耦数据边界与物理画布边界
    # 显式拓宽真实的 X 轴可见范围，将两侧原本悬空的排版文本重新纳入 Axes 内部
    ax.set_xlim(0.55, 1.15) 
    
    # 仅在真实数据的有效跨度区间 (0.75-0.95) 渲染 X 轴刻度
    ax.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95])
    
    # Y 轴坐标基准
    y_positions = np.arange(len(df))
    ax.set_ylim(-1, len(df))
    
    # 提取 Overall 均值作为垂直虚线参考
    overall_auroc = df[df['Category'] == 'Overall']['AUROC'].values[0]
    ax.axvline(x=overall_auroc, color=COLOR_REF, linestyle='--', linewidth=2, zorder=0, alpha=0.8)

    # 遍历数据逐行渲染
    current_category = ""
    for i, row in df.iterrows():
        y = y_positions[i]
        
        # 修改后：
        # 1. 绘制左侧文本 (亚组分类名称)
        if row['Category'] != current_category and row['Category'] != 'Overall':
            # 收缩向左的偏移量，保持排版紧凑且不重叠
            ax.text(x_min - 0.14, y + 0.4, row['Category'], ha='left', va='center', fontweight='bold', fontsize=12)
            current_category = row['Category']
        
        # 打印具体的亚组名和样本量
        indent = "" if row['Category'] == 'Overall' else "  "
        font_w = 'bold' if row['Category'] == 'Overall' else 'normal'
        ax.text(x_min - 0.14, y, f"{indent}{row['Subgroup']} (n={row['N']})", ha='left', va='center', fontweight=font_w, fontsize=11)
        
        # 2. 绘制中间的森林图 (点和误差线)
        # 误差线
        ax.hlines(y, row['CI_L'], row['CI_U'], color=COLOR_LINE, linewidth=2.5, zorder=1)
        # 均值点 (Overall 用菱形，其他用圆点)
        marker_style = 'D' if row['Category'] == 'Overall' else 'o'
        marker_size = 100 if row['Category'] == 'Overall' else 80
        ax.scatter(row['AUROC'], y, color=COLOR_DOT, edgecolor='black', linewidth=1, 
                   s=marker_size, marker=marker_style, zorder=2)
        
        # 修改后：
        # 3. 绘制右侧文本 (AUROC 数值与 P-value)
        auroc_text = f"{row['AUROC']:.3f} ({row['CI_L']:.3f}-{row['CI_U']:.3f})"
        ax.text(x_max + 0.02, y, auroc_text, ha='left', va='center', fontweight=font_w, fontsize=11)
        
        if row['P_val']:
            # 【修复】将偏移量从 0.12 扩大至 0.18，为左侧的置信区间留出充足物理空间
            ax.text(x_max + 0.14, y, row['P_val'], ha='center', va='center', fontsize=11)
    # 修改后：
    # -----------------------------------------------------------------
    # 表头渲染 (Headers)
    # -----------------------------------------------------------------
    header_y = len(df)
    ax.text(x_min - 0.14, header_y, "Subgroup", ha='left', va='center', fontweight='bold', fontsize=12)
    ax.text(x_max + 0.02, header_y, "AUROC (95% CI)", ha='left', va='center', fontweight='bold', fontsize=12)
    ax.text(x_max + 0.18, header_y, "P for interaction", ha='center', va='center', fontweight='bold', fontsize=12)
    
    # 绘制表头下的分割线 (收缩横线的物理跨度，使其精确贴合并闭合两端文字，且强制不被裁剪)
    ax.hlines(y=header_y - 0.5, xmin=x_min - 0.14, xmax=x_max + 0.16, color='black', linewidth=1.5, clip_on=False)
    ax.hlines(y=-0.5, xmin=x_min - 0.14, xmax=x_max + 0.16, color='black', linewidth=1.5, clip_on=False)

    # -----------------------------------------------------------------
    # 坐标轴与边界清理
    # -----------------------------------------------------------------
    ax.set_yticks([])
    ax.set_xlabel("Area Under the Receiver Operating Characteristic Curve (AUROC)", fontweight='bold', fontsize=12, labelpad=10)
    
    # 仅保留底部边框，剔除上方、左侧、右侧边框
    sns.despine(ax=ax, left=True, right=True, top=True)
    
    # 标题定制：修改为强制居中对齐 (loc='center')
    # plt.title('Subgroup AUROC Forest Plot Demonstrating Model Robustness', 
    #           loc='center', fontweight='bold', fontsize=16, pad=30)

    # -----------------------------------------------------------------
    # 导出高规图像
    # -----------------------------------------------------------------
    plt.tight_layout()
    out_path_png = os.path.join(fig_dir, "Fig_1c_Subgroup_Forest_Plot.png")
    out_path_pdf = os.path.join(fig_dir, "Fig_1c_Subgroup_Forest_Plot.pdf")
    
    # 保存时扩展 bbox_inches 确保图外的文本不被裁剪
    plt.savefig(out_path_png, dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig(out_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n🎉 完美！Figure 1c 临床亚组森林图渲染完毕！(400 DPI)")
    print(f"👉 查阅路径: {out_path_png}")

if __name__ == "__main__":
    generate_forest_plot()

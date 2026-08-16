"""
DILI-PLUS | 旧版张量人工核查脚本（已归档）

职责：解码旧版 tensors/ 目录中的用药、时间和诊断张量，抽样生成病历级核查报告。
输入：X_med.pt、T_med.pt、X_diag.pt、Y_aki.pt、Y_dili.pt 及对应词表。
输出：reports/04_Clinical_Sanity_Check_Report.md。
状态：遗留脚本；依赖当前项目中不存在的旧版 AKI/DILI 联合张量，不属于现行
Parquet + DILIPlusDataset 单任务主流程。保留用于追溯，当前不应作为流水线步骤运行。
"""

import os
import torch
import json
import numpy as np
from datetime import datetime
import random

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return {v: k for k, v in vocab.items()}

def decode_med_sequence(x_tensor, t_tensor, id2med):
    """解码药物序列并计算真实时间 (修复时间维度爆炸 Bug)"""
    decoded = []
    times = []
    
    for i in range(len(x_tensor)):
        token_id = x_tensor[i].item()
        if token_id == 0:  # <PAD>
            continue
            
        token_str = id2med.get(token_id, "<UNK>")
        
        # 🌟 核心修复: T_med.pt 里面存放的就是原始的绝对小时数，直接读取即可
        t_original = t_tensor[i].item() 
        times.append(t_original)
        
        decoded.append(f"[{t_original:>6.1f} h] {token_str}")
        
    time_span = max(times) - min(times) if times else 0.0
    return decoded, time_span, len(decoded)

def decode_diag_sequence(x_tensor, id2diag):
    """解码诊断序列"""
    decoded = []
    for i in range(len(x_tensor)):
        token_id = x_tensor[i].item()
        if token_id == 0:
            continue
        decoded.append(id2diag.get(token_id, "<UNK>"))
    return decoded

def run_sanity_check_and_report():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tensor_dir = os.path.join(base_dir, "tensors")
    vocab_dir = os.path.join(base_dir, "vocab")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    report_path = os.path.join(reports_dir, "04_Clinical_Sanity_Check_Report.md")
    
    print("🔍 Loading Tensors and Vocabularies...")
    X_med = torch.load(os.path.join(tensor_dir, "X_med.pt"))
    T_med = torch.load(os.path.join(tensor_dir, "T_med.pt"))
    X_diag = torch.load(os.path.join(tensor_dir, "X_diag.pt"))
    Y_aki = torch.load(os.path.join(tensor_dir, "Y_aki.pt"))
    Y_dili = torch.load(os.path.join(tensor_dir, "Y_dili.pt"))
    
    id2med = load_vocab(os.path.join(vocab_dir, "vocab_polypharmacy.json"))
    id2diag = load_vocab(os.path.join(vocab_dir, "vocab_diagnosis.json"))
    
    # 抽取索引
    aki_pos_idx = (Y_aki == 1).nonzero(as_tuple=True)[0].tolist()
    dili_pos_idx = (Y_dili == 1).nonzero(as_tuple=True)[0].tolist()
    neg_idx = ((Y_aki == 0) & (Y_dili == 0)).nonzero(as_tuple=True)[0].tolist()
    
    # 随机种子保证每次抽取不同，或者固定种子方便复现
    random.seed(42) 
    sample_aki = random.sample(aki_pos_idx, min(5, len(aki_pos_idx)))
    sample_dili = random.sample(dili_pos_idx, min(5, len(dili_pos_idx)))
    sample_neg = random.sample(neg_idx, min(5, len(neg_idx)))
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 🏥 医疗多模态张量逆向核查报告 (Clinical Sanity Check)\n\n")
        f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("> **核查目的:** 验证多模态张量在时间轴密集度、因果基线洗脱以及药代动力学上的临床合理性，为 SCI 论文的严谨性提供病历级证据。\n\n")
        
        def write_case(idx, case_type):
            f.write(f"### 📌 {case_type} (Tensor Index: `{idx}`)\n")
            f.write(f"- **Labels:** AKI = `{Y_aki[idx].item()}` | DILI = `{Y_dili[idx].item()}`\n")
            
            # 诊断
            diags = decode_diag_sequence(X_diag[idx], id2diag)
            f.write(f"- **[生理基线] 既往诊断 ({len(diags)} 项):**\n")
            if diags:
                f.write(f"  > {', '.join(diags)}\n")
            else:
                f.write(f"  > *无既往诊断记录*\n")
            
            # 用药
            meds, time_span, med_count = decode_med_sequence(X_med[idx], T_med[idx], id2med)
            f.write(f"- **[干预窗] 用药轨迹:**\n")
            f.write(f"  - 总给药频次: `{med_count}` 次\n")
            f.write(f"  - 绝对时间跨度: `{time_span:.1f}` 小时 ({time_span/24:.1f} 天)\n")
            f.write("  ```text\n")
            for m in meds:
                f.write(f"  {m}\n")
            f.write("  ```\n\n")
            f.write("---\n\n")

        f.write("## 1. 🔥 AKI 阳性高危病例抽样 (AKI = 1)\n")
        for idx in sample_aki: write_case(idx, "典型急性肾损伤")
            
        f.write("## 2. ☢️ DILI 阳性高危病例抽样 (DILI = 1)\n")
        for idx in sample_dili: write_case(idx, "典型药物性肝损伤")
            
        f.write("## 3. 🛡️ 双阴性安全病例抽样 (AKI = 0, DILI = 0)\n")
        for idx in sample_neg: write_case(idx, "安全多重用药")

    print(f"\n✅ Sanity Check Report successfully generated at:\n📁 {report_path}")
    print("👉 请打开该 Markdown 文件，并将其全部内容发送给我。我们将联合进行临床级架构审查。")

if __name__ == "__main__":
    run_sanity_check_and_report()

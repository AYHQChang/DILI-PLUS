"""
DILI-PLUS | 旧版张量样本打印工具（已归档）

职责：从旧版 tensors/ 目录随机选择 AKI 阳性样本，解码用药、时间和诊断张量。
输入：X_med.pt、T_med.pt、X_diag.pt、Y_aki.pt、Y_dili.pt 及词表。
输出：终端中的单病例张量明细。
状态：遗留脚本；当前项目的 tensors/ 为空，且主任务已改为单一 DILI 标签，
因此不属于现行 Parquet + DILIPlusDataset 流程。
"""

import os
import torch
import numpy as np
import json
import random

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def visualize_sample():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tensor_dir = os.path.join(base_dir, "tensors")
    vocab_dir = os.path.join(base_dir, "vocab")
    
    print("🚀 Loading multi-modal tensors and vocabularies...")
    X_med = torch.load(os.path.join(tensor_dir, "X_med.pt"))
    T_med = torch.load(os.path.join(tensor_dir, "T_med.pt"))
    X_diag = torch.load(os.path.join(tensor_dir, "X_diag.pt"))
    Y_aki = torch.load(os.path.join(tensor_dir, "Y_aki.pt"))
    Y_dili = torch.load(os.path.join(tensor_dir, "Y_dili.pt"))
    
    vocab_med = load_vocab(os.path.join(vocab_dir, "vocab_polypharmacy.json"))
    id2med = {v: k for k, v in vocab_med.items()}
    vocab_diag = load_vocab(os.path.join(vocab_dir, "vocab_diagnosis.json"))
    id2diag = {v: k for k, v in vocab_diag.items()}
    
    # 随机寻找一个 AKI 阳性患者
    aki_pos_idx = (Y_aki == 1).nonzero(as_tuple=True)[0].tolist()
    target_idx = random.choice(aki_pos_idx)
    
    x_m = X_med[target_idx]
    t_m = T_med[target_idx]
    x_d = X_diag[target_idx]
    
    print("\n" + "█"*60)
    print(f" 🔍 COMPLETE DATA SAMPLE VISUALIZATION (Index: {target_idx})")
    print("█"*60)
    
    print(f"\n🎯 [TARGET LABELS]")
    print(f"   ► AKI (Acute Kidney Injury) Occurred:  {'YES' if Y_aki[target_idx].item() == 1 else 'NO'}")
    print(f"   ► DILI (Drug-Induced Liver Injury) Occurred: {'YES' if Y_dili[target_idx].item() == 1 else 'NO'}")
    
    print(f"\n🧠 [MODALITY 2: STATIC BASELINE (X_diag)]")
    diags = []
    for i in range(len(x_d)):
        if x_d[i].item() == 0: continue
        diags.append(id2diag.get(x_d[i].item(), "<UNK>"))
    print(f"   ► 既往诊断记录 ({len(diags)} 项):")
    if diags:
        print(f"     " + " | ".join(diags))
    else:
        print("     [无诊断记录]")
        
    print(f"\n💊 [MODALITY 1: DYNAMIC SEQUENCES (X_med & T_med)]")
    print(f"   ► 有效用药频次: {(x_m != 0).sum().item()} / 160 (Max Len)")
    print(f"   {'Token_ID':<10} | {'Absolute Time':<15} | {'Log-Scaled Time':<15} | {'Medication Name'}")
    print("   " + "-"*75)
    
    for i in range(len(x_m)):
        if x_m[i].item() == 0: continue
        token_id = x_m[i].item()
        med_name = id2med.get(token_id, "<UNK>")
        abs_time = t_m[i].item()
        log_time = np.log1p(abs_time) # 模拟网络内的对数平滑
        
        print(f"   {token_id:<10} | [{abs_time:>6.1f} h]       | [{log_time:>6.3f}]        | {med_name}")

    print("\n" + "█"*60 + "\n")

if __name__ == "__main__":
    visualize_sample()

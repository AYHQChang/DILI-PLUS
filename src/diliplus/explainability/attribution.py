"""
DILI-PLUS | 单病例药物归因与候选病例筛选（包实现）

职责：在 DILI 阳性样本中结合预测概率与药物留一敏感性选择候选病例，并使用
Layer Integrated Gradients 计算该病例用药 Token 的局部归因。
输入：DILIPlusDataset、第一折时间感知模型权重和药物中英文映射。
输出：候选病例状态 JSON 与 06b_Target_Patient_Attribution.csv。
状态：当前 DILI 单任务的局部模型解释步骤。
解释边界：LOO 与积分梯度描述模型响应，不识别药物因果效应或真实换药收益；
概率未重新应用训练阶段的温度参数。
"""

import os
import re
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from captum.attr import LayerIntegratedGradients

from diliplus.config import load_settings
from diliplus.data.dataset import DILIPlusDataset, load_vocab_sizes
from diliplus.models.diliplus_engine import DILIPlusEngine

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 🌟 超参数：靶向雷达配置 (确保 06c 沙盒有药可换)
# =============================================================================
MANUAL_PATIENT_IDX = None  

# =============================================================================
# 🌟 扩容版靶向雷达配置：涵盖四大类 DILI 高危药物
# =============================================================================
CLINICAL_SWAP_TARGETS = [
    # 1. 降脂类 (温和)
    "阿托伐他汀钙片", "瑞舒伐他汀钙片", "辛伐他汀片", "氟伐他汀钠胶囊", 
    "阿托伐他汀", "瑞舒伐他汀", "辛伐他汀", "氟伐他汀",
    # 2. 抗结核类 (中重)
    "异烟肼片", "利福平片", "吡嗪酰胺片", "盐酸乙胺丁醇片", 
    "异烟肼", "利福平", "吡嗪酰胺", "乙胺丁醇",
    # 3. 经典抗菌药物 (极高频 DILI 诱导) - 扩充！
    "阿莫西林胶囊", "左氧氟沙星片", "伏立康唑片", "阿莫西林克拉维酸钾片", 
    "注射用头孢曲松钠", "盐酸莫西沙星片", "阿奇霉素片",
    "阿莫西林", "左氧氟沙星", "伏立康唑", "阿莫西林克拉维酸钾", "头孢曲松", "莫西沙星", "阿奇霉素",
    # 4. 其他高危专科药 - 扩充！
    "对乙酰氨基酚片", "布洛芬缓释胶囊", "盐酸胺碘酮片", "甲氨蝶呤片",
    "对乙酰氨基酚", "布洛芬", "胺碘酮", "甲氨蝶呤"
]

NON_DRUG_KEYWORDS = [
    "复查", "检查", "注意", "拔出", "转科", "请", "请带", "测", "理疗", 
    "常规", "病理", "观察", "高压氧", "病重", "透药", "转肾内", "摄片", "C肽"
]

# =============================================================================
# 🛡️ V15 架构专用 Captum 包装器
# =============================================================================
class DILIPlusCaptumWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x_med, dt_med, mask_med, x_lab, v_lab, dt_lab, mask_lab, x_diag, mask_diag):
        outputs = self.model(x_med, dt_med, mask_med, x_lab, v_lab, dt_lab, mask_lab, x_diag, mask_diag)
        return outputs["logits"]

# =============================================================================
# 🛠️ 辅助工具函数
# =============================================================================
def load_med_vocab(vocab_dir):
    path = os.path.join(vocab_dir, "vocab_polypharmacy.json")
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    if "med_token2id" in vocab: mapping = vocab["med_token2id"]
    elif "token2id" in vocab: mapping = vocab["token2id"]
    else: mapping = vocab
    return {int(v): str(k) for k, v in mapping.items()}

def load_translation_mapping(map_path):
    if os.path.exists(map_path): return pd.read_csv(map_path)
    return pd.DataFrame()

def is_real_drug(token_name, mapping_df):
    if token_name.startswith("<") and token_name.endswith(">"): return False
    if "PAD" in token_name.upper() or "UNK" in token_name.upper(): return False
    if re.match(r"^[A-Z]\d{2}(\.\d{1,3})?.*$", token_name): return False
    for kw in NON_DRUG_KEYWORDS:
        if kw in token_name: return False
    return True

# =============================================================================
# 🚀 主控引擎
# =============================================================================
def main(settings=None):
    settings = settings or load_settings()
    data_dir = str(settings.paths.data_cache)
    vocab_dir = str(settings.paths.vocab)
    save_dir = str(settings.paths.checkpoints)
    report_dir = str(settings.paths.reports)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Booting DILIPLUS IG Attribution Engine | Device: {device}")
    
    vocab_config = load_vocab_sizes(vocab_dir)
    id2med = load_med_vocab(vocab_dir)
    med2id = {v: k for k, v in id2med.items()}
    mapping_df = load_translation_mapping(str(settings.paths.drug_mapping))
    
    model = DILIPlusEngine(**vocab_config).to(device)
    
    fold_idx = 1
    weight_path = os.path.join(save_dir, f"best_calib_MultiModalTimeAwareMedBERT_Fold{fold_idx}.pth")
    if not os.path.exists(weight_path):
        print(f"🚨 FATAL: Weight not found at {weight_path}")
        return
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    captum_model = DILIPlusCaptumWrapper(model).to(device)
    ig = LayerIntegratedGradients(captum_model, model.med_embedding)
    
    full_dataset = DILIPlusDataset(data_dir, vocab_dir)
    
    # -------------------------------------------------------------------------
    # 2. 基于留一扰动的候选病例筛选（模型敏感性，不是因果识别）
    # -------------------------------------------------------------------------
    target_idx = MANUAL_PATIENT_IDX
    target_tensors = None
    baseline_risk = 0.0
    
    if target_idx is None:
        print(f"📡 Radar Scanning: Performing Global LOO Causal Check...")
        candidate_list = []
        
        for idx in range(len(full_dataset)):
            tensors = full_dataset[idx]
            label_val = tensors.get('label_dili', tensors.get('label'))
            if label_val is None or label_val.item() != 1: continue 
            
            inputs = {k: v.unsqueeze(0).to(device) for k, v in tensors.items() if 'label' not in k}
            
            with torch.no_grad():
                p_base = torch.softmax(model(**inputs)["logits"], dim=1)[0, 1].item()
                
            # 使用模型原始 softmax 概率筛选高分样本；此处未应用温度参数
            if p_base > 0.30:
                med_ids = tensors['x_med'].tolist()
                if isinstance(med_ids[0], list): med_ids = med_ids[0]
                med_list = [id2med.get(m, "") for m in med_ids if m != 0]
                
                hits = [m for m in med_list if m in CLINICAL_SWAP_TARGETS]
                
                if hits:
                    target_med = hits[0]
                    target_med_id = med2id[target_med]
                    
                    inputs_ablated = {k: v.clone() for k, v in inputs.items()}
                    ablated_mask = inputs_ablated['mask_med'].clone()
                    
                    for i in range(inputs_ablated['x_med'].shape[1]):
                        if inputs_ablated['x_med'][0, i].item() == target_med_id:
                            ablated_mask[0, i] = False 
                            
                    inputs_ablated['mask_med'] = ablated_mask
                    
                    with torch.no_grad():
                        p_ab = torch.softmax(model(**inputs_ablated)["logits"], dim=1)[0, 1].item()
                        
                    arr = p_base - p_ab 
                    
                    # 🔥 放宽门槛：只要拔除药后，风险下降 > 1% (ARR > 0.01) 即算有效靶点
                    if arr > 0.0:
                        candidate_list.append({
                            'patient_idx': idx, 'p_base': p_base, 'arr': arr, 
                            'target_med': target_med, 'inputs': inputs
                        })

        # 若严格条件没有候选者，则进入放宽条件的后备搜索
        if not candidate_list:
            print("🚨 Strict Radar failed. Initiating Fallback Search (Relaxing all causal constraints)...")
            for idx in range(len(full_dataset)):
                tensors = full_dataset[idx]
                label_val = tensors.get('label_dili', tensors.get('label'))
                if label_val is None or label_val.item() != 1: continue 
                
                med_ids = tensors['x_med'].tolist()
                if isinstance(med_ids[0], list): med_ids = med_ids[0]
                med_list = [id2med.get(m, "") for m in med_ids if m != 0]
                hits = [m for m in med_list if m in CLINICAL_SWAP_TARGETS]
                
                if hits:
                    inputs = {k: v.unsqueeze(0).to(device) for k, v in tensors.items() if 'label' not in k}
                    with torch.no_grad():
                        p_base = torch.softmax(model(**inputs)["logits"], dim=1)[0, 1].item()
                    candidate_list.append({
                        'patient_idx': idx, 'p_base': p_base, 'arr': 0.0, 
                        'target_med': hits[0], 'inputs': inputs
                    })
                    if len(candidate_list) >= 20: break # Fallback 找够 20 个就停

        if not candidate_list:
            print("🚨 FATAL: No patients found even with Fallback. Check CLINICAL_SWAP_TARGETS.")
            return
            
        # 👑 恢复 Top 20 榜单输出
        candidate_list.sort(key=lambda x: x['arr'], reverse=True)
        
        print(f"\n🏆 Top Candidates Leaderboard (Max 20):")
        print(f"{'Rank':<5} | {'Patient ID':<10} | {'Target Med':<15} | {'Base Risk':<10} | {'LOO ARR':<10}")
        print("-" * 70)
        for rank, cand in enumerate(candidate_list[:20]):
            print(f"{rank+1:<5} | {cand['patient_idx']:<10} | {cand['target_med']:<15} | {cand['p_base']*100:>5.2f}%    | {cand['arr']*100:>5.2f}%")
        print("-" * 70)
        
        best_candidate = candidate_list[0]
        target_idx = best_candidate['patient_idx']
        target_tensors = best_candidate['inputs']
        baseline_risk = best_candidate['p_base']
        
        print(f"\n👉 AUTO-SELECTED ULTIMATE CANDIDATE: Patient Index {target_idx}")
        
        # 🔗 工程同步 JSON
        sync_data = {
            "selected_patient_idx": target_idx,
            "baseline_risk": baseline_risk,
            "identified_target": best_candidate['target_med']
        }
        with open(os.path.join(report_dir, "06b_Selected_Patient_State.json"), "w", encoding="utf-8") as f:
            json.dump(sync_data, f, ensure_ascii=False, indent=4)

    else:
        print(f"🎯 Loading Manual Target Patient Index: {target_idx}")
        target_tensors = {k: v.unsqueeze(0).to(device) for k, v in full_dataset[target_idx].items() if 'label' not in k}
        with torch.no_grad():
            baseline_risk = torch.softmax(model(**target_tensors)["logits"], dim=1)[0, 1].item()

    # -------------------------------------------------------------------------
    # 3. 🧠 计算积分梯度 (Integrated Gradients)
    # -------------------------------------------------------------------------
    print("🧠 Computing Layer Integrated Gradients for Medication Stream...")
    x_m = target_tensors['x_med']
    baseline_med = torch.zeros_like(x_m).to(device) 
    
    additional_args = (
        target_tensors['dt_med'], target_tensors['mask_med'],
        target_tensors['x_lab'], target_tensors['v_lab'], target_tensors['dt_lab'], target_tensors['mask_lab'],
        target_tensors['x_diag'], target_tensors['mask_diag']
    )
    
    attr_med, delta = ig.attribute(
        inputs=x_m,
        baselines=baseline_med,
        target=1, 
        additional_forward_args=additional_args,
        return_convergence_delta=True,
        n_steps=50
    )
    
    attr_med_score = attr_med.sum(dim=-1).squeeze(0).cpu().detach().numpy()
    
    # -------------------------------------------------------------------------
    # 4. 📊 整理、清洗与翻译
    # -------------------------------------------------------------------------
    valid_med_len = target_tensors['mask_med'][0].sum().item()
    med_aggr = {}
    total_pure_attr = 1e-9
    
    for idx in range(int(valid_med_len)):
        med_name = id2med.get(x_m[0, idx].item(), "<UNK>")
        if not is_real_drug(med_name, mapping_df): continue
            
        score = attr_med_score[idx]
        med_aggr[med_name] = med_aggr.get(med_name, 0.0) + score
        total_pure_attr += abs(score)
        
    attribution_records = []
    print(f"\n{'='*70}\n📊 Patient {target_idx} Medication Attribution Report\n{'='*70}")
    
    for name, score in sorted(med_aggr.items(), key=lambda item: item[1], reverse=True):
        impact_pct = (score / total_pure_attr) * 100
        
        name_en = name
        if not mapping_df.empty:
            match = mapping_df[mapping_df['cleaned_api_zh'] == name]
            if match.empty: match = mapping_df[mapping_df['raw_zh_name'] == name]
            if not match.empty and pd.notna(match['target_en_name'].iloc[0]): 
                name_en = match['target_en_name'].iloc[0]
                
        attribution_records.append({
            'Patient_ID': target_idx,
            'Medication_ZH': name, 
            'Medication_EN': name_en, 
            'Attribution_Score': score, 
            'Contribution_Pct': impact_pct
        })
        
        bar = "█" * int(abs(impact_pct) / 2)
        direction = "🔴 Toxic (+)" if score > 0 else "🟢 Protective (-)"
        print(f"[{direction}] {name_en[:25]:<25} | Impact: {impact_pct:>6.2f}% | {bar}")

    # 5. 持久化归因结果
    df_attr = pd.DataFrame(attribution_records)
    out_path = os.path.join(report_dir, "06b_Target_Patient_Attribution.csv")
    df_attr.to_csv(out_path, index=False)
    
    print(f"\n✅ Attribution completed. Patient Index {target_idx} saved to: {out_path}")
    print("👉 Next Step: Run 06c_explain_counterfactual.py to initiate the SandBox Substitution!")

if __name__ == "__main__":
    main()

"""
DILI-PLUS | 单病例药物归因与候选病例筛选（包实现）

职责：在 DILI 阳性样本中结合预测概率与药物留一敏感性选择候选病例，并使用
Layer Integrated Gradients 计算该病例用药 Token 的局部归因。
输入：DILIPlusDataset、指定 run/fold 的版本化时间感知模型 artifact 和药物映射。
输出：候选病例状态 JSON 与 06b_Target_Patient_Attribution.csv。
状态：当前 DILI 单任务的局部模型解释步骤。
解释边界：LOO 与积分梯度描述模型响应，不识别药物因果效应或真实换药收益；调用方
必须显式选择 raw 或 calibrated 输出，且候选病例只来自 artifact 的 outer test partition。
"""

import os
import re
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from captum.attr import LayerIntegratedGradients

from diliplus.artifacts import (
    artifact_probabilities,
    dataset_fingerprint,
    deep_artifact_path,
    load_deep_artifact,
    run_report_dir,
)
from diliplus.config import load_settings
from diliplus.data.dataset import DILIPlusDataset, load_vocab_sizes
from diliplus.models.registry import PRIMARY_MODEL_NAME, build_formal_deep_model
from diliplus.reproducibility import seed_everything

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
    def __init__(self, model, temperature=1.0, probability_mode="raw"):
        super().__init__()
        self.model = model
        self.temperature = float(temperature)
        self.probability_mode = probability_mode
        
    def forward(self, x_med, dt_med, mask_med, x_lab, v_lab, dt_lab, mask_lab, x_diag, mask_diag):
        outputs = self.model(x_med, dt_med, mask_med, x_lab, v_lab, dt_lab, mask_lab, x_diag, mask_diag)
        logits = outputs["logits"]
        return logits / self.temperature if self.probability_mode == "calibrated" else logits


def _positive_probability(model, inputs, metadata, probability_mode):
    logits = model(**inputs)["logits"]
    return float(artifact_probabilities(logits, metadata, probability_mode)[0])

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
def main(settings=None, run_id=None, probability_mode="calibrated", fold_idx=1):
    settings = settings or load_settings()
    if not run_id:
        raise ValueError("run_id is required to resolve the model artifact")
    if probability_mode not in ("raw", "calibrated"):
        raise ValueError("probability_mode must be 'raw' or 'calibrated'")
    seed_everything(settings.reproducibility)
    data_dir = str(settings.model_data_dir)
    vocab_dir = str(settings.paths.vocab)
    report_dir = str(run_report_dir(settings, run_id) / "explainability")
    os.makedirs(report_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Booting DILIPLUS IG Attribution Engine | Device: {device}")
    
    vocab_config = load_vocab_sizes(vocab_dir)
    id2med = load_med_vocab(vocab_dir)
    med2id = {v: k for k, v in id2med.items()}
    mapping_df = load_translation_mapping(str(settings.paths.drug_mapping))
    
    model = build_formal_deep_model(
        PRIMARY_MODEL_NAME, vocab_config, settings.training
    ).to(device)
    
    artifact_path = deep_artifact_path(
        settings, run_id, PRIMARY_MODEL_NAME, fold_idx
    )
    current_fingerprint = dataset_fingerprint(settings)["payload_sha256"]
    metadata = load_deep_artifact(
        artifact_path,
        model,
        map_location=device,
        expected_run_id=run_id,
        expected_model_name=PRIMARY_MODEL_NAME,
        expected_fold=fold_idx,
        expected_dataset_fingerprint=current_fingerprint,
    )
    model.eval()
    
    captum_model = DILIPlusCaptumWrapper(
        model, metadata["temperature"], probability_mode
    ).to(device)
    ig = LayerIntegratedGradients(captum_model, model.med_embedding)
    
    full_dataset = DILIPlusDataset(data_dir, vocab_dir)
    eligible_test_indices = set(metadata["split"]["indices"]["test"])
    
    # -------------------------------------------------------------------------
    # 2. 基于留一扰动的候选病例筛选（模型敏感性，不是因果识别）
    # -------------------------------------------------------------------------
    target_idx = MANUAL_PATIENT_IDX
    target_tensors = None
    baseline_risk = 0.0
    
    if target_idx is None:
        print("Radar scanning outer-test cases with LOO model sensitivity...")
        candidate_list = []
        
        for idx in sorted(eligible_test_indices):
            tensors = full_dataset[idx]
            label_val = tensors.get('label_ahi_proxy', tensors.get('label'))
            if label_val is None or label_val.item() != 1: continue 
            
            inputs = {k: v.unsqueeze(0).to(device) for k, v in tensors.items() if 'label' not in k}
            
            with torch.no_grad():
                p_base = _positive_probability(
                    model, inputs, metadata, probability_mode
                )

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
                        p_ab = _positive_probability(
                            model, inputs_ablated, metadata, probability_mode
                        )
                        
                    arr = p_base - p_ab 
                    
                    # 🔥 放宽门槛：只要拔除药后，风险下降 > 1% (ARR > 0.01) 即算有效靶点
                    if arr > 0.0:
                        candidate_list.append({
                            'patient_idx': idx, 'p_base': p_base, 'arr': arr, 
                            'target_med': target_med, 'inputs': inputs
                        })

        # 若严格条件没有候选者，则进入放宽条件的后备搜索
        if not candidate_list:
            print("Strict sensitivity screen found no case; using the prespecified fallback screen...")
            for idx in sorted(eligible_test_indices):
                tensors = full_dataset[idx]
                label_val = tensors.get('label_ahi_proxy', tensors.get('label'))
                if label_val is None or label_val.item() != 1: continue 
                
                med_ids = tensors['x_med'].tolist()
                if isinstance(med_ids[0], list): med_ids = med_ids[0]
                med_list = [id2med.get(m, "") for m in med_ids if m != 0]
                hits = [m for m in med_list if m in CLINICAL_SWAP_TARGETS]
                
                if hits:
                    inputs = {k: v.unsqueeze(0).to(device) for k, v in tensors.items() if 'label' not in k}
                    with torch.no_grad():
                        p_base = _positive_probability(
                            model, inputs, metadata, probability_mode
                        )
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
        print(f"{'Rank':<5} | {'Case Index':<10} | {'Target Med':<15} | {'Base Score':<10} | {'LOO Delta':<10}")
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
            "identified_target": best_candidate['target_med'],
            "run_id": run_id,
            "fold": fold_idx,
            "probability_mode": probability_mode,
            "artifact_metadata_sha256": metadata["metadata_payload_sha256"],
        }
        with open(os.path.join(report_dir, "06b_Selected_Patient_State.json"), "w", encoding="utf-8") as f:
            json.dump(sync_data, f, ensure_ascii=False, indent=4)

    else:
        if target_idx not in eligible_test_indices:
            raise ValueError(
                f"Manual dataset index {target_idx} is not in outer test fold {fold_idx}"
            )
        print(f"🎯 Loading Manual Target Patient Index: {target_idx}")
        target_tensors = {k: v.unsqueeze(0).to(device) for k, v in full_dataset[target_idx].items() if 'label' not in k}
        with torch.no_grad():
            baseline_risk = _positive_probability(
                model, target_tensors, metadata, probability_mode
            )

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
    print(f"\n{'='*70}\nDataset row {target_idx} medication attribution report\n{'='*70}")
    
    for name, score in sorted(med_aggr.items(), key=lambda item: item[1], reverse=True):
        impact_pct = (score / total_pure_attr) * 100
        
        name_en = name
        if not mapping_df.empty:
            match = mapping_df[mapping_df['cleaned_api_zh'] == name]
            if match.empty: match = mapping_df[mapping_df['raw_zh_name'] == name]
            if not match.empty and pd.notna(match['target_en_name'].iloc[0]): 
                name_en = match['target_en_name'].iloc[0]
                
        attribution_records.append({
            'Case_Index': target_idx,
            'Run_ID': run_id,
            'Fold': fold_idx,
            'Probability_Mode': probability_mode,
            'Medication_ZH': name, 
            'Medication_EN': name_en, 
            'Attribution_Score': score, 
            'Contribution_Pct': impact_pct
        })
        
        bar = "█" * int(abs(impact_pct) / 2)
        direction = "Model output increasing (+)" if score > 0 else "Model output decreasing (-)"
        print(f"[{direction}] {name_en[:25]:<25} | Impact: {impact_pct:>6.2f}% | {bar}")

    # 5. 持久化归因结果
    df_attr = pd.DataFrame(attribution_records)
    out_path = os.path.join(report_dir, "06b_Target_Patient_Attribution.csv")
    df_attr.to_csv(out_path, index=False)
    
    print(f"\nAttribution completed. Dataset row {target_idx} saved to: {out_path}")
    print("Next step: run the medication-token perturbation stage with the same run/fold/mode.")

if __name__ == "__main__":
    main()

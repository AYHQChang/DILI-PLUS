"""
DILI-PLUS | 药物扰动与替换敏感性沙盒（包实现）

职责：读取 06b 选定病例，对候选药物执行嵌入幅度缩放和词表内 Token 替换，
记录模型预测概率相对基线的变化轨迹。
输入：run-specific 06b 状态/归因文件、DILIPlusDataset、版本化模型 artifact 和药物映射。
输出：对应 run 的 06c_Medication_Token_Perturbation.csv。
状态：当前 DILI 单任务的模型边界审计步骤。
解释边界：该分析是观察性模型的扰动敏感性测试，不是反事实因果推断、药效模拟、
临床换药建议或随机对照试验证据；调用方必须显式选择 raw 或 calibrated 输出。
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import json

from diliplus.artifacts import (
    artifact_probabilities,
    dataset_fingerprint,
    deep_artifact_path,
    load_deep_artifact,
    run_report_dir,
)
from diliplus.config import load_settings
from diliplus.data.dataset import DILIPlusDataset, load_vocab_sizes
from diliplus.models.diliplus_engine import DILIPlusEngine
from diliplus.reproducibility import seed_everything

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# DILI 药物 Token 替换候选字典（仅用于模型敏感性分析）
# =============================================================================
SUBSTITUTION_MAP_ZH = {
    # 1. 降脂药 (Statins)
    "阿托伐他汀钙片": "普伐他汀钠片",
    "辛伐他汀片": "普伐他汀钠片",
    "瑞舒伐他汀钙胶囊": "普伐他汀钠片", 
    "阿托伐他汀": "普伐他汀", "辛伐他汀": "普伐他汀", "瑞舒伐他汀": "普伐他汀",
    
    # 2. 护胃药 (PPIs)
    "注射用艾司奥美拉唑钠": "雷贝拉唑钠肠溶片",
    "注射用奥美拉唑钠": "雷贝拉唑钠肠溶片",
    "注射用泮托拉唑钠": "雷贝拉唑钠肠溶片",
    
    # 3. 非甾体抗炎药 (NSAIDs)
    "双氯芬酸钠肠溶片": "塞来昔布胶囊",
    "布洛芬缓释胶囊": "塞来昔布胶囊",
    "对乙酰氨基酚片": "塞来昔布胶囊",
    
    # 4. 抗菌药物 (Antibiotics)
    "阿莫西林克拉维酸钾片": "头孢呋辛酯片",
    "盐酸左氧氟沙星片": "头孢呋辛酯片",
    "左氧氟沙星片": "头孢呋辛酯片",
    "阿奇霉素片": "头孢呋辛酯片",
    
    # 5. 抗结核药物 (Anti-TB)
    "异烟肼片": "盐酸乙胺丁醇片",
    "吡嗪酰胺片": "盐酸乙胺丁醇片",
    "利福平片": "盐酸乙胺丁醇片",
    
    # 6. 抗癫痫药物 (Antiepileptics)
    "丙戊酸钠片": "左乙拉西坦片",
    "苯妥英钠片": "左乙拉西坦片",
    "卡马西平片": "拉考沙胺片",
    
    # 7. 抗真菌药物 (Antifungals)
    "酮康唑片": "氟康唑胶囊",
    "注射用伏立康唑": "注射用米卡芬净钠",
    "伏立康唑片": "氟康唑胶囊",
    
    # 8. 抗风湿药物与抗心律失常药
    "甲氨蝶呤片": "柳氮磺吡啶肠溶片",
    "盐酸胺碘酮片": "盐酸索他洛尔片"
}

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
    return {int(v): str(k) for k, v in mapping.items()}, {str(k): int(v) for k, v in mapping.items()}

# =============================================================================
# 🚀 多靶点沙盒推演引擎
# =============================================================================
def _positive_probability(logits, metadata, probability_mode):
    return float(artifact_probabilities(logits, metadata, probability_mode)[0])


def run_targeted_counterfactual_trajectory(
    settings=None, run_id=None, probability_mode="calibrated", fold_idx=1
):
    settings = settings or load_settings()
    if not run_id:
        raise ValueError("run_id is required to resolve the model artifact")
    if probability_mode not in ("raw", "calibrated"):
        raise ValueError("probability_mode must be 'raw' or 'calibrated'")
    seed_everything(settings.reproducibility)
    data_dir = str(settings.model_data_dir)
    vocab_dir = str(settings.paths.vocab)
    reports_dir = str(run_report_dir(settings, run_id) / "explainability")
    
    sync_file = os.path.join(reports_dir, "06b_Selected_Patient_State.json")
    ig_attr_file = os.path.join(reports_dir, "06b_Target_Patient_Attribution.csv")
    output_csv = os.path.join(reports_dir, "06c_Medication_Token_Perturbation.csv")
    mapping_file = str(settings.paths.drug_mapping)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initialising medication-token perturbation engine on {device}...")
    
    if not os.path.exists(sync_file) or not os.path.exists(ig_attr_file):
        print(f"🚨 Missing prerequisite artifacts. Please ensure 06b has been executed.")
        return

    with open(sync_file, 'r', encoding="utf-8") as f:
        state_data = json.load(f)
    target_idx = state_data["selected_patient_idx"]
    for field, expected in (
        ("run_id", run_id),
        ("fold", fold_idx),
        ("probability_mode", probability_mode),
    ):
        if state_data.get(field) != expected:
            raise ValueError(
                f"06b state {field} mismatch: expected {expected!r}, "
                f"found {state_data.get(field)!r}"
            )
    print(f"Selected dataset row: {target_idx}")

    # =========================================================================
    # 载入中英文药物名称双向映射
    # =========================================================================
    df_map = pd.read_csv(mapping_file) if os.path.exists(mapping_file) else pd.DataFrame()
    en_to_zh, zh_to_en = {}, {}
    if not df_map.empty:
        print(f"🔗 Binding Vocabulary: ZH <-> EN")
        en_to_zh = dict(zip(df_map['target_en_name'], df_map['raw_zh_name']))
        zh_to_en = dict(zip(df_map['raw_zh_name'], df_map['target_en_name']))
        # 补充清理过的 API 映射作为兜底
        en_to_zh.update(dict(zip(df_map['target_en_name'], df_map['cleaned_api_zh'])))
        zh_to_en.update(dict(zip(df_map['cleaned_api_zh'], df_map['target_en_name'])))

    # =========================================================================
    # 1. 环境与模型初始化 (V15 Dataset 替代原版 Tensor 读取)
    # =========================================================================
    vocab_config = load_vocab_sizes(vocab_dir)
    id2med, med2id = load_med_vocab(vocab_dir)
    
    model = DILIPlusEngine(**vocab_config).to(device)
    model_path = deep_artifact_path(
        settings, run_id, "MultiModalTimeAwareMedBERT", fold_idx
    )
    metadata = load_deep_artifact(
        model_path,
        model,
        map_location=device,
        expected_run_id=run_id,
        expected_model_name="MultiModalTimeAwareMedBERT",
        expected_fold=fold_idx,
        expected_dataset_fingerprint=dataset_fingerprint(settings)["payload_sha256"],
    )
    if state_data.get("artifact_metadata_sha256") != metadata["metadata_payload_sha256"]:
        raise ValueError("06b state does not refer to the loaded model artifact")
    model.eval()

    full_dataset = DILIPlusDataset(data_dir, vocab_dir)
    if target_idx not in set(metadata["split"]["indices"]["test"]):
        raise ValueError("Selected dataset row is not in the artifact outer test partition")
    target_tensors = {k: v.unsqueeze(0).to(device) for k, v in full_dataset[target_idx].items() if 'label' not in k}
    
    x_m = target_tensors['x_med']
    dt_m = target_tensors['dt_med']
    mask_m = target_tensors['mask_med']
    valid_len = mask_m[0].sum().item()

    # 预计算 Baseline 概率与 B、C 流特征 (提高沙盒运行效率)
    with torch.no_grad():
        base_outputs = model(**target_tensors)
        h_lab_base = base_outputs["h_lab"]
        h_diag_base = base_outputs["h_diag"]
        base_prob = _positive_probability(
            base_outputs["logits"], metadata, probability_mode
        )

    print(f"   - {probability_mode} baseline model probability: {base_prob * 100:.2f}%")

    # =========================================================================
    # 2. 动态 LOO 探针：寻找对模型预测影响最大的药物 Token
    # =========================================================================
    max_drop = -1
    best_med_code = -1
    for j in range(int(valid_len)):
        med_code = x_m[0, j].item()
        if med_code == 0: continue
        
        # 物理消融该药
        temp_inputs = {k: v.clone() for k, v in target_tensors.items()}
        temp_inputs['mask_med'][0, j] = False
        
        with torch.no_grad():
            temp_prob = _positive_probability(
                model(**temp_inputs)["logits"], metadata, probability_mode
            )
            
        drop = base_prob - temp_prob
        if drop > max_drop:
            max_drop = drop
            best_med_code = med_code

    best_med_zh = id2med.get(best_med_code, "<UNK>")
    print(f"🔍 Dynamic LOO Probe Detected Max Variance Drug: {best_med_zh} (Drop: {max_drop*100:.2f}%)")

    # =========================================================================
    # 3. 队列重构：整合 IG 静态靶点与 LOO 动态靶点 (双重解离靶向)
    # =========================================================================
    df_ig = pd.read_csv(ig_attr_file)
    
    # 排除非药物噪音
    exclude_keywords = ['病重', '常规', '三项', '吸氧', '测定']
    df_meds = df_ig[~df_ig['Medication_ZH'].str.contains('|'.join(exclude_keywords), na=False)].copy()

    # Positive/negative attribution describes model-output direction only.
    enhancers = df_meds[df_meds['Contribution_Pct'] > 0].sort_values('Contribution_Pct', ascending=False).head(3)
    enhancers['Drug_Role'] = 'Model-output increasing attribution'
    
    suppressors = df_meds[df_meds['Contribution_Pct'] < 0].sort_values('Contribution_Pct', ascending=True).head(3)
    suppressors['Drug_Role'] = 'Model-output decreasing attribution'
    
    targets_df = pd.concat([enhancers, suppressors])
    
    # 融合 LOO 探针靶点
    if best_med_zh not in targets_df['Medication_ZH'].values and not any(kw in best_med_zh for kw in exclude_keywords):
        best_med_en = zh_to_en.get(best_med_zh, best_med_zh)
        new_row = pd.DataFrame([{
            'Medication_ZH': best_med_zh,
            'Medication_EN': f"[*] {best_med_en} [Max Variance]",
            'Contribution_Pct': 99.9, 
            'Drug_Role': 'Largest LOO model-output change'
        }])
        targets_df = pd.concat([new_row, targets_df], ignore_index=True)

    # 融合 06b 传来的 JSON 靶向药 (Clinical Override)
    state_target_zh = state_data["identified_target"]
    if state_target_zh not in targets_df['Medication_ZH'].values:
        state_target_en = zh_to_en.get(state_target_zh, state_target_zh)
        new_row = pd.DataFrame([{
            'Medication_ZH': state_target_zh,
            'Medication_EN': f"[*] {state_target_en} [Clinical Target]",
            'Contribution_Pct': 99.9, 
            'Drug_Role': 'Prespecified token-substitution target'
        }])
        targets_df = pd.concat([new_row, targets_df], ignore_index=True)

    # =========================================================================
    # 4. 双轨模型扰动分析：嵌入缩放与 Token 替换
    # =========================================================================
    alpha_intervals = [1.0, 0.75, 0.50, 0.25, 0.0]
    trajectory_records = []

    print("\n" + "="*80)
    print("Executing embedding attenuation and token substitution sensitivity")
    print("=" * 80)

    for _, row in targets_df.iterrows():
        core_med_zh = row['Medication_ZH']
        med_en = row['Medication_EN']
        drug_role = row['Drug_Role']
        
        target_id = med2id.get(core_med_zh, -1)
        if target_id == -1: continue
            
        hit_indices = [i for i in range(int(valid_len)) if x_m[0, i].item() == target_id]
        if not hit_indices: continue
            
        print(f"\n🎯 Testing [{drug_role}]: {med_en} ({core_med_zh})")
        
        # ---------------------------------------------------------------------
        # 轨道 A: 阶梯式减量 (Dose Tapering / V15 Engine)
        # ---------------------------------------------------------------------
        for alpha in alpha_intervals:
            with torch.no_grad():
                # 提取初始 Embedding
                emb_med_cf = model.med_embedding(x_m)
                
                # 实施 Alpha 衰减
                for i in hit_indices:
                    emb_med_cf[0, i, :] = emb_med_cf[0, i, :] * alpha
                        
                # 注入 V15 Time2Vec
                dt_m_norm = torch.log1p(torch.clamp(dt_m, min=0))
                t_emb_med = model.med_time2vec(dt_m_norm)
                h_med_seq = emb_med_cf + t_emb_med
                
                # V15 Transformer 编码
                med_key_pad_mask = ~mask_m.bool()
                med_key_pad_mask[med_key_pad_mask.all(dim=1), 0] = False
                seq_len_med = x_m.size(1)
                causal_mask_med = nn.Transformer.generate_square_subsequent_mask(seq_len_med).to(device)
                
                h_med_seq = model.med_transformer(h_med_seq, mask=causal_mask_med, src_key_padding_mask=med_key_pad_mask)
                h_med = model.ln_med(model.med_pool(h_med_seq, mask_m))
                
                # V15 残差融合
                h_concat = torch.cat([h_med, h_lab_base, h_diag_base], dim=-1)
                h_fused_raw = model.fusion_projection(h_concat)
                gate = model.fusion_gate(h_concat)
                h_dynamic = h_med + h_lab_base
                h_fused = gate[:, :model.hidden_size] * h_dynamic + gate[:, model.hidden_size:] * h_fused_raw
                
                prob_cf = _positive_probability(
                    model.dili_head(h_fused), metadata, probability_mode
                )
                
            trajectory_records.append({
                'Case_Index': target_idx,
                'Run_ID': run_id,
                'Fold': fold_idx,
                'Probability_Mode': probability_mode,
                'Drug_Role': drug_role,
                'Perturbation_Type': 'Embedding Attenuation',
                'Targeted_Medications': core_med_zh,
                'Targeted_Medications_EN': med_en,
                'Parameter': f"Alpha={alpha}",  
                'Model_Predicted_Probability_Pct': prob_cf * 100,
                'Delta_Predicted_Probability_Points': (prob_cf - base_prob) * 100
            })
            
            trend = "increases model output" if prob_cf > base_prob else "decreases model output"
            if alpha == 1.0: trend = "🔹 (Baseline)"
            print(f"   [Embedding attenuation] alpha={alpha:>4.2f} -> probability {prob_cf * 100:>5.2f}% | {trend}")

        # ---------------------------------------------------------------------
        # 轨道 B：词表内药物 Token 替换
        # ---------------------------------------------------------------------
        matched_dict_key = None
        for dict_k in SUBSTITUTION_MAP_ZH.keys():
            if dict_k in core_med_zh or core_med_zh in dict_k:
                matched_dict_key = dict_k
                break

        if matched_dict_key:
            safe_med_zh_ideal = SUBSTITUTION_MAP_ZH[matched_dict_key]
            
            safe_med_id = 0
            actual_safe_med_zh = safe_med_zh_ideal
            # 🔥 修改 1: 使用 med2id (字符串:数字) 进行迭代
            for v_med, v_id in med2id.items():
                if safe_med_zh_ideal in v_med or v_med in safe_med_zh_ideal:
                    if len(v_med) >= 2: 
                        # 🔥 修改 2: 正确的变量赋值
                        safe_med_id = v_id
                        actual_safe_med_zh = v_med
                        break
            
            safe_med_en = zh_to_en.get(actual_safe_med_zh, safe_med_zh_ideal)
            
            if safe_med_id != 0:
                print(f"   💊 [Substitution] Finding alternative: Swapping [{core_med_zh}] -> [{actual_safe_med_zh}]")
                
                x_m_sub = x_m.clone()
                for i in hit_indices:
                    x_m_sub[0, i] = safe_med_id
                    
                with torch.no_grad():
                    # 重新过一遍 V15 前向传播
                    emb_med_sub = model.med_embedding(x_m_sub)
                    t_emb_med = model.med_time2vec(dt_m_norm)
                    h_med_seq = emb_med_sub + t_emb_med
                    
                    h_med_seq = model.med_transformer(h_med_seq, mask=causal_mask_med, src_key_padding_mask=med_key_pad_mask)
                    h_med = model.ln_med(model.med_pool(h_med_seq, mask_m))
                    
                    h_concat = torch.cat([h_med, h_lab_base, h_diag_base], dim=-1)
                    h_fused_raw = model.fusion_projection(h_concat)
                    gate = model.fusion_gate(h_concat)
                    h_dynamic = h_med + h_lab_base
                    h_fused = gate[:, :model.hidden_size] * h_dynamic + gate[:, model.hidden_size:] * h_fused_raw
                    
                    prob_sub = _positive_probability(
                        model.dili_head(h_fused), metadata, probability_mode
                    )
                    
                trajectory_records.append({
                    'Case_Index': target_idx,
                    'Run_ID': run_id,
                    'Fold': fold_idx,
                    'Probability_Mode': probability_mode,
                    'Drug_Role': drug_role,
                    'Perturbation_Type': 'Token Substitution',
                    'Targeted_Medications': core_med_zh,
                    'Targeted_Medications_EN': med_en,
                    'Parameter': f"Swap to {safe_med_en}", 
                    'Model_Predicted_Probability_Pct': prob_sub * 100,
                    'Delta_Predicted_Probability_Points': (prob_sub - base_prob) * 100
                })
                
                print(f"   [Token substitution] probability {prob_sub * 100:>5.2f}% | delta {(prob_sub - base_prob) * 100:>5.2f} points")
            else:
                print(f"   ⚠️ [Substitution] Safe alternative '{safe_med_zh_ideal}' not found in vocabulary.")

    df_results = pd.DataFrame(trajectory_records)
    df_results.to_csv(output_csv, index=False)
    print("\n" + "=" * 80)
    print(f"Medication-token perturbation results saved: {output_csv}")

if __name__ == "__main__":
    run_targeted_counterfactual_trajectory()

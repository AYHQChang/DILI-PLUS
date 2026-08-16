"""
DILI-PLUS | 旧版药物扰动脚本副本（已归档）

职责：保留早期多目标药物嵌入缩放与 Token 替换实现，供历史结果追溯。
输入：旧版模型模块、旧版模型权重及 06b 结果。
输出：旧版扰动轨迹 CSV。
状态：遗留副本；依赖当前仓库中不存在的 models.multimodal_medbert，不属于现行主流程。
解释边界：脚本中的“因果/反事实”旧称仅表示模型输入扰动，不能解释为临床因果效应。
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd
import json
from models.multimodal_medbert import MultiModalTimeAwareMedBERT

# =============================================================================
# 🌟 DILI 反事实替代字典 (Counterfactual Substitution Ontology)
# 将高危肝毒性药物映射为理论上同类但肝毒性较低的替代药物。
# HQ，请根据您的临床常识或药学专家建议修改此字典。
# =============================================================================
# =============================================================================
# 🌟 DILI 反事实替代字典 (Counterfactual Substitution Ontology)
# 核心配对逻辑: [治疗等效性 (Therapeutic Equivalence)] + [药代动力学肝脏解耦 (Pharmacokinetic Divergence)]
# =============================================================================
# SUBSTITUTION_MAP_ZH = {
#     # 1. 降脂药 (Statins)
#     # 机制：瑞舒伐他汀/阿托伐他汀为脂溶性，依赖肝脏酶系；普伐他汀为水溶性，极少经 CYP450 代谢，肝毒性极低。
#     "瑞舒伐他汀钙胶囊": "普伐他汀钠片",
#     "阿托伐他汀钙片": "普伐他汀钠片",
    
#     # 2. 质子泵抑制剂 (PPIs / 护胃药)
#     # 机制：奥美拉唑强烈依赖/抑制肝脏 CYP2C19 酶，极易引发药物交互肝损伤；泮托拉唑对肝酶亲和力最低，肝功能不全首选。
#     "注射用艾司奥美拉唑钠": "注射用泮托拉唑钠",
#     "注射用奥美拉唑钠": "注射用泮托拉唑钠",
    
#     # 3. 抗菌药 (Antibiotics)
#     # 机制：阿莫西林克拉维酸钾是全球范围内导致特异质型 DILI (Idiosyncratic DILI) 排名第一的元凶；头孢类（头孢呋辛）引发肝损的概率极低。
#     "注射用阿莫西林钠克拉维酸钾": "注射用头孢呋辛钠",
#     "阿莫西林克拉维酸钾分散片": "注射用头孢呋辛钠",
    
#     # 4. 抗血小板药 (Antiplatelets)
#     # 机制：替格瑞洛在某些队列中观察到较高的无症状转氨酶升高；氯吡格雷相对更为经典且肝脏耐受性更好（作为对比实验）。
#     "替格瑞洛片": "硫酸氢氯吡格雷片（波立维）",
    
#     # 5. 抗结核药 (Anti-Tuberculosis)
#     # 机制：利福平/异烟肼是公认的经典肝毒性药物；乙胺丁醇在抗结核一线药物中不具有肝毒性。
#     "利福平注射液": "盐酸乙胺丁醇片",
#     "异烟肼片": "盐酸乙胺丁醇片"
# }
SUBSTITUTION_MAP_ZH = {
    # ---------------------------------------------------------
    # 1. 降脂药 (Statins): 亲脂性高代谢负担 -> 亲水性低CYP依赖转移
    # 循证依据: 阿托伐他汀和辛伐他汀高度依赖CYP3A4，DILI报告发生率及死亡信号(ROR=2.96-3.09)在同类中极高。
    # 普伐他汀为高度水溶性分子，几乎不经过CYP3A4代谢，在DILIN及7.1万人荟萃分析中具有极其优异的安全记录。
    # ---------------------------------------------------------
    "阿托伐他汀钙片": "普伐他汀钠片",
    "辛伐他汀片": "普伐他汀钠片",
    "瑞舒伐他汀钙胶囊": "普伐他汀钠片", # 瑞舒虽肝毒性仅为中等，但在极端器官保护推演中，考虑其肾脏/肌肉风险，建议降级替换。
    
    # ---------------------------------------------------------
    # 2. 护胃药 (PPIs): CYP2C19不可逆抑制剂 -> 非酶促降解途径优化 (重大错误修正)
    # 严正修正: 原拟定为泮托拉唑是极其危险的。泮托拉唑在CTP C级肝病患者中暴露量激增4-8倍，被国际指南列为“Unsafe”。
    # 雷贝拉唑约50%依赖非酶促化学途径降解，受CYP多态性变异影响极小，被列为肝硬化合并症的优先安全用药。
    # ---------------------------------------------------------
    "注射用艾司奥美拉唑钠": "雷贝拉唑钠肠溶片",
    "注射用奥美拉唑钠": "雷贝拉唑钠肠溶片",
    "注射用泮托拉唑钠": "雷贝拉唑钠肠溶片",
    
    # ---------------------------------------------------------
    # 3. 非甾体抗炎药 (NSAIDs): 高特异质肝毒性中间体 -> 安全性验证的COX-2抑制剂
    # 循证依据: 大型临床研究(包含2.4万人的多项RCT)及前瞻性DILIN研究明确指出，双氯芬酸具有同类中最强的致肝损伤概率(肝酶升高率达4.2%)。
    # 塞来昔布转氨酶大幅升高的比率(1.1%)与安慰剂对照组(0.9%)相比几乎无统计学差异，无临床显性肝炎报告。
    # ---------------------------------------------------------
    "双氯芬酸钠肠溶片": "塞来昔布胶囊",
    "布洛芬缓释胶囊": "塞来昔布胶囊",
    
    # ---------------------------------------------------------
    # 4. 抗菌药物 (Antibiotics): DILI流行病学绝对致病主因 -> 低风险二代头孢的规避替代
    # 循证依据: 阿莫西林克拉维酸钾是全球导致DILI住院(特异质型胆汁淤积损伤)的第一大单药因素，相对风险极高。
    # 头孢呋辛酯在头对头随机对照试验中，导致肝脏及全身不良反应的发生率(18%)显著低于阿莫西林克拉维酸钾组(39%)，极少发生4+级肝损。
    # 左氧氟沙星的肝损住院调整后相对风险高达3.2倍，应予替换。
    # ---------------------------------------------------------
    "阿莫西林克拉维酸钾片": "头孢呋辛酯片",
    "盐酸左氧氟沙星片": "头孢呋辛酯片",
    "阿奇霉素片": "头孢呋辛酯片",
    
    # ---------------------------------------------------------
    # 5. 抗结核药物 (Anti-TB): 暴发性肝衰竭药物组合 -> 非肝毒性一线治疗药物解救方案
    # 循证依据: 异烟肼、利福平与吡嗪酰胺(HRZE)是诱发致死性急性肝衰竭(LiverTox最高危5+级)的最强协同组合。
    # 乙胺丁醇的代谢和毒性靶点主要集中于视神经，完全剥离了对肝脏生化反应的干涉，是肝酶严重异常患者抢救性重症结核病首选基石药物。
    # ---------------------------------------------------------
    "异烟肼片": "盐酸乙胺丁醇片",
    "吡嗪酰胺片": "盐酸乙胺丁醇片",
    
    # ---------------------------------------------------------
    # 6. 抗癫痫药物 (Antiepileptics): 强肝脏代谢及微泡脂肪变性 -> 肾脏清除/线性无相互作用PK模式
    # 循证依据: 丙戊酸和苯妥英易诱发急剧氧化应激爆发、致命性高氨血症及SJS等重症免疫型肝反应(LiverTox分类中归为A/B类极高危)。
    # 左乙拉西坦及拉考沙胺(归为C/D类安全级)极少经肝实质细胞代谢，产生无活性代谢物，不干扰微粒体酶系统，被视为伴发严重肝损的完美用药。
    # ---------------------------------------------------------
    "丙戊酸钠片": "左乙拉西坦片",
    "苯妥英钠片": "左乙拉西坦片",
    "卡马西平片": "拉考沙胺片",
    
    # ---------------------------------------------------------
    # 7. 抗真菌药物 (Antifungals): 强线粒体功能毒性损伤 -> 轻度一过性自限反应与靶点分离
    # 循证依据: 酮康唑和伏立康唑具有证实直接破坏细胞电子传递链的线粒体毒性，其转氨酶急剧上升发生率在大型分析中逼近20%。
    # 氟康唑导致的大多为停药即刻可逆的一过性轻度异常；而米卡芬净作为靶向细胞壁的棘白菌素类，在临床中将肝不良事件导致的停药风险降低了超过50%。
    # ---------------------------------------------------------
    "酮康唑片": "氟康唑胶囊",
    "注射用伏立康唑": "注射用米卡芬净钠",
    
    # ---------------------------------------------------------
    # 8. 抗风湿药物与抗心律失常药: 特殊专科替代防线
    # 循证依据: 甲氨蝶呤大剂量或持续使用具备促肝脏纤维化的高危风险，针对肝损人群建议换用柳氮磺吡啶。
    # 胺碘酮具有约1/4100的严重显性急性肝损发生率，对于合并肝炎病史的严重心律失常可考虑改用索他洛尔以平衡安全性。
    # ---------------------------------------------------------
    "甲氨蝶呤片": "柳氮磺吡啶肠溶片",
    "盐酸胺碘酮片": "盐酸索他洛尔片"
}

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_targeted_counterfactual_trajectory():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tensor_dir = os.path.join(base_dir, "tensors")
    vocab_dir = os.path.join(base_dir, "vocab")
    reports_dir = os.path.join(base_dir, "reports")
    
    sync_file = os.path.join(reports_dir, "selected_patient_idx.txt")
    ig_attr_file = os.path.join(reports_dir, "06b_IG_Attribution.csv")
    output_csv = os.path.join(reports_dir, "06c_Counterfactual_Trajectory.csv")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initialising Multi-Target Counterfactual Engine on {device}...")
    
    model_name = "MultiModalTimeAwareMedBERT"
    report_csv = os.path.join(reports_dir, "05_Experiment_Results_Table.csv")
    
    try:
        df_perf = pd.read_csv(report_csv)
        best_fold_idx = int(df_perf[df_perf['Model_Architecture'] == model_name]
                            .sort_values(by='DILI_AUPRC', ascending=False)
                            .iloc[0]['Fold'])
    except (FileNotFoundError, IndexError):
        print(f"\n⚠️ Performance logs not found for {model_name}, defaulting to Fold 1.")
        best_fold_idx = 1
        
    model_path = os.path.join(base_dir, "saved_models", f"best_{model_name}_Fold{best_fold_idx}.pth")

    if not os.path.exists(sync_file) or not os.path.exists(ig_attr_file):
        print(f"🚨 Missing prerequisite artifacts. Please ensure 06b has been executed.")
        return

    with open(sync_file, 'r') as f:
        target_idx = int(f.read().strip())
    print(f"🎯 Precision Target Locked: Patient Index {target_idx}")

    # =========================================================================
    # 1. 环境与模型初始化
    # =========================================================================
    X_med = torch.load(os.path.join(tensor_dir, "X_med.pt"))
    T_med = torch.load(os.path.join(tensor_dir, "T_med.pt"))
    X_diag = torch.load(os.path.join(tensor_dir, "X_diag.pt"))
    
    vocab_med = load_vocab(os.path.join(vocab_dir, "vocab_polypharmacy.json"))
    id2med = {v: k for k, v in vocab_med.items()}
    
    vocab_poly_size = max(vocab_med.values()) + 1
    vocab_diag_size = 5000
    try:
        with open(os.path.join(vocab_dir, "vocab_diagnosis.json"), "r", encoding="utf-8") as f:
            vocab_diag_size = max(json.load(f).values()) + 1
    except Exception:
        pass

    model = MultiModalTimeAwareMedBERT(vocab_poly_size=vocab_poly_size, vocab_diag_size=vocab_diag_size).to(device)
    
    if not os.path.exists(model_path):
        print(f"🚨 Model checkpoint not found at {model_path}")
        return
        
    print(f"🟢 Initialising Causal Sandbox for {model_name} (Using Best Fold: {best_fold_idx})...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x_m = X_med[target_idx].unsqueeze(0).to(device)
    t_m = T_med[target_idx].unsqueeze(0).to(device)
    x_d = X_diag[target_idx].unsqueeze(0).to(device)
    
    med_mask = (x_m != 0).long()
    diag_mask = (x_d != 0).long()
    valid_len = med_mask.sum().item()
    t_m_log = torch.log1p(t_m)

    # =========================================================================
    # 2. 🌟 动态 LOO 探针：强制捕获最大药代动力学方差药物 (切换为 DILI 头)
    # =========================================================================
    with torch.no_grad():
        _, logits_base = model(x_m, t_m_log, med_mask, x_d, diag_mask)
        base_prob = F.softmax(logits_base, dim=1)[0, 1].item()

    max_drop = -1
    best_med_code = -1
    for j in range(valid_len):
        med_code = x_m[0, j].item()
        if med_code == 0: continue
        
        temp_x_m = x_m.clone()
        temp_x_m[0, j] = 0
        temp_med_mask = (temp_x_m != 0).long()
        
        with torch.no_grad():
            _, temp_logits = model(temp_x_m, t_m_log, temp_med_mask, x_d, diag_mask)
            temp_prob = F.softmax(temp_logits, dim=1)[0, 1].item()
            
        drop = base_prob - temp_prob
        if drop > max_drop:
            max_drop = drop
            best_med_code = med_code

    best_med_zh = id2med.get(best_med_code, "<UNK>")
    print(f"🔍 Dynamic LOO Probe Detected Max Variance Drug: {best_med_zh} (Drop: {max_drop*100:.2f}%)")

    # =========================================================================
    # 3. 队列重构：整合 IG 静态靶点与 LOO 动态靶点
    # =========================================================================
    df_ig = pd.read_csv(ig_attr_file)
    df_meds = df_ig[df_ig['Type'] == 'Medication'].copy()
    
    # 排除非药物噪音
    exclude_keywords = ['病重', '常规', '三项', '吸氧', '测定']
    df_meds = df_meds[~df_meds['Feature'].str.contains('|'.join(exclude_keywords), na=False)]

    enhancers = df_meds[df_meds['Contribution_Pct'] > 0].sort_values('Contribution_Pct', ascending=False).head(3)
    enhancers['Drug_Role'] = 'Risk Enhancer (Pathogenic)'
    
    suppressors = df_meds[df_meds['Contribution_Pct'] < 0].sort_values('Contribution_Pct', ascending=True).head(3)
    suppressors['Drug_Role'] = 'Risk Suppressor (Protective)'
    
    targets_df = pd.concat([enhancers, suppressors])
    
    if best_med_zh not in targets_df['Feature'].values and not any(kw in best_med_zh for kw in exclude_keywords):
        en_match = df_meds[df_meds['Feature'] == best_med_zh]
        best_med_en = en_match['Feature_EN'].iloc[0] if not en_match.empty else best_med_zh
        best_med_en = f"[*] {best_med_en} [Max Variance]"
        
        new_row = pd.DataFrame([{
            'Feature': best_med_zh,
            'Feature_EN': best_med_en,
            'Type': 'Medication',
            'Contribution_Pct': 99.9, 
            'Drug_Role': 'Risk Enhancer (Pathogenic)'
        }])
        targets_df = pd.concat([new_row, targets_df], ignore_index=True)

    # =========================================================================
    # 4. 🌟 双轨反事实沙盘推演: Dose Tapering & Counterfactual Substitution
    # =========================================================================
    alpha_intervals = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    trajectory_records = []

    print("\n" + "-"*80)
    print("⏳ Executing Counterfactual Interventions: Tapering & Substitution")
    print("-" * 80)

    with torch.no_grad():
        emb_med_base = model.dynamic_encoder.token_embedding(x_m) + model.dynamic_encoder.time_embedding(t_m_log.unsqueeze(-1))
        emb_diag_base = model.static_encoder.diag_embedding(x_d)

    for _, row in targets_df.iterrows():
        med_zh = row['Feature']
        med_en = row['Feature_EN']
        drug_role = row['Drug_Role']
        
        hit_indices = [i for i in range(valid_len) if id2med.get(x_m[0, i].item(), "<UNK>") == med_zh]
        if not hit_indices: continue
            
        print(f"\n🧪 Testing [{drug_role}]: {med_en} ({med_zh})")
        baseline_risk = 0.0
        
        # ---------------------------------------------------------------------
        # 轨道 A: 阶梯式减量 (Dose Tapering / Withdrawal)
        # ---------------------------------------------------------------------
        for alpha in alpha_intervals:
            emb_med_cf = emb_med_base.clone() 
            
            for i in hit_indices:
                emb_med_cf[0, i, :] = emb_med_cf[0, i, :] * alpha
                    
            with torch.no_grad():
                encoded_seq = model.dynamic_encoder.transformer_encoder(emb_med_cf, src_key_padding_mask=(med_mask == 0))
                mask_m_exp = med_mask.unsqueeze(-1).expand_as(encoded_seq).float()
                h_dyn = model.ln_dynamic(torch.sum(encoded_seq * mask_m_exp, dim=1) / torch.clamp(mask_m_exp.sum(dim=1), min=1e-9))
                
                mask_d_exp = diag_mask.unsqueeze(-1).expand_as(emb_diag_base).float()
                h_stat = model.ln_static(model.static_encoder.projection(torch.sum(emb_diag_base * mask_d_exp, dim=1) / torch.clamp(mask_d_exp.sum(dim=1), min=1e-9)))
                
                h_concat = torch.cat([h_dyn, h_stat], dim=-1)
                h_fused_raw = model.fusion_projection(h_concat)
                gate = model.fusion_gate(h_concat)
                h_fused = gate[:, :128] * h_dyn + gate[:, 128:] * h_fused_raw
                
                logits_cf = model.dili_head(h_fused)
                prob_cf = F.softmax(logits_cf, dim=1)[0, 1].item()
                
            if alpha == 1.0:
                baseline_risk = prob_cf * 100
                
            trajectory_records.append({
                'Patient_ID': target_idx,  
                'Drug_Role': drug_role,
                'Intervention_Type': 'Dose Tapering',
                'Targeted_Medications': med_zh,
                'Targeted_Medications_EN': med_en,
                'Parameter': f"Alpha={alpha}",  
                'Predicted_DILI_Risk': prob_cf * 100, 
                'Absolute_Risk_Reduction': baseline_risk - (prob_cf * 100)
            })
            
            trend = "🔺 (Escalation)" if prob_cf * 100 > baseline_risk else "🔻 (Reduction)"
            if alpha == 1.0: trend = "🔹 (Baseline)"
            print(f"   [Tapering] Presence Alpha: {alpha:>3.1f} -> DILI Risk: {prob_cf * 100:>5.2f}% | Trend: {trend}")

        # ---------------------------------------------------------------------
        # 轨道 B: 🌟 反事实药物替代 (Counterfactual Substitution / Token Swap)
        # ---------------------------------------------------------------------
        if med_zh in SUBSTITUTION_MAP_ZH:
            safe_med_zh = SUBSTITUTION_MAP_ZH[med_zh]
            safe_med_id = vocab_med.get(safe_med_zh, 0)
            
            if safe_med_id != 0:
                print(f"   💊 [Substitution] Swapping [{med_zh}] with [{safe_med_zh}]")
                
                # 直接在 Token 序列张量上进行硬替换
                x_m_cf = x_m.clone()
                for i in hit_indices:
                    x_m_cf[0, i] = safe_med_id
                    
                with torch.no_grad():
                    # 因为 Token 变了，必须重新查表获取新的 Base Embedding
                    emb_med_sub = model.dynamic_encoder.token_embedding(x_m_cf) + model.dynamic_encoder.time_embedding(t_m_log.unsqueeze(-1))
                    
                    encoded_seq = model.dynamic_encoder.transformer_encoder(emb_med_sub, src_key_padding_mask=(med_mask == 0))
                    h_dyn = model.ln_dynamic(torch.sum(encoded_seq * mask_m_exp, dim=1) / torch.clamp(mask_m_exp.sum(dim=1), min=1e-9))
                    
                    h_concat = torch.cat([h_dyn, h_stat], dim=-1)
                    h_fused_raw = model.fusion_projection(h_concat)
                    gate = model.fusion_gate(h_concat)
                    h_fused = gate[:, :128] * h_dyn + gate[:, 128:] * h_fused_raw
                    
                    logits_sub = model.dili_head(h_fused)
                    prob_sub = F.softmax(logits_sub, dim=1)[0, 1].item()
                    
                trajectory_records.append({
                    'Patient_ID': target_idx,  
                    'Drug_Role': drug_role,
                    'Intervention_Type': 'Substitution',
                    'Targeted_Medications': med_zh,
                    'Targeted_Medications_EN': f"Swapped to: {safe_med_zh}",
                    'Parameter': f"Swap Token",  
                    'Predicted_DILI_Risk': prob_sub * 100, 
                    'Absolute_Risk_Reduction': baseline_risk - (prob_sub * 100)
                })
                
                print(f"   [Substitution] Safe Alternative Risk -> DILI Risk: {prob_sub * 100:>5.2f}% | 🔻 ARR: {baseline_risk - (prob_sub * 100):>5.2f}%")
            else:
                print(f"   ⚠️ [Substitution] Safe alternative '{safe_med_zh}' not found in vocabulary. Skipping swap.")

    df_results = pd.DataFrame(trajectory_records)
    df_results.to_csv(output_csv, index=False)

    print("\n" + "-" * 80)
    print(f"🎉 Multi-Target Trajectories Exported: {output_csv}")

if __name__ == "__main__":
    run_targeted_counterfactual_trajectory()

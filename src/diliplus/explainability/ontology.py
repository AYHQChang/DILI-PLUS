"""
DILI-PLUS | 双语药物映射与替换候选表构建（包实现）

职责：从动态词表生成药物映射模板，并根据已完成的中英文/ATC 映射，为同 ATC
大类且同给药途径的药物建立候选替换列表。
输入：vocab/vocab_polypharmacy.json、mapping_completed_drugs_checkpoint.csv。
输出：mapping_template_drugs.csv、data_cache/Safety_Substitution_Map.json。
状态：解释性分析的辅助工具；实际依赖 04 生成的词表，因此编号不代表真实执行顺序。
解释边界：候选表仅用于模型敏感性测试，不代表临床等效或因果安全替代建议。
"""

import os
import json
import pandas as pd
import re
from pathlib import Path

from diliplus.config import load_settings

def build_polypharmacy_mapping_template(vocab_path, output_csv):
    if not os.path.exists(vocab_path):
        print(f"⚠️ [DILIPLUS] Vocab missing: {vocab_path} (Skipping template generation)")
        return

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    rows = []
    for raw_name, token_id in vocab.items():
        if token_id < 4: continue
        route = "IV" if any(x in raw_name for x in ["注射", "输液"]) else "PO"
        clean_name = re.sub(r'\(.*?\)|注射液|肠溶片|片|胶囊|分散片|注射用|口服液', '', raw_name).strip()
        
        rows.append({
            "token_id": token_id, "raw_zh_name": raw_name, 
            "cleaned_api_zh": clean_name, "route": route,
            "target_en_name": "", "atc_code": ""        
        })
        
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ [DILIPLUS] Template exported: {output_csv}")

def generate_safety_substitution_map(completed_mapping_csv, output_json):
    """
    DILIPLUS Track B 核心：构建安全替换映射图谱。
    逻辑：寻找同 ATC 类别 (前3位相同) 但肝毒性较低的同途径 (route) 替代药物。
    这里构建基础框架，后续可以在沙盒分析时动态调用。
    """
    if not os.path.exists(completed_mapping_csv):
        print(f"⚠️ [DILIPLUS] Completed mapping CSV not found: {completed_mapping_csv}")
        print("请确保已完成 LLM 翻译。")
        return
        
    df = pd.read_csv(completed_mapping_csv)
    
    # 过滤出有合法 ATC code 的药物
    df_valid = df[df['atc_code'].notna() & (df['atc_code'] != '')].copy()
    df_valid['atc_core'] = df_valid['atc_code'].astype(str).str[:3] # 取 ATC 前三位作为药理大类
    
    substitution_map = {}
    
    # 构建同类药物池
    for _, row in df.iterrows():
        token_id = str(row['token_id'])
        if pd.isna(row['atc_code']) or len(str(row['atc_code'])) < 3:
            substitution_map[token_id] = [] # 无明确大类的，不提供替换
            continue
            
        atc_core = str(row['atc_code'])[:3]
        route = row['route']
        
        # 寻找同大类、同给药途径的“其他”药物作为候选池
        candidates = df_valid[
            (df_valid['atc_core'] == atc_core) & 
            (df_valid['route'] == route) & 
            (df_valid['token_id'] != row['token_id'])
        ]['token_id'].astype(str).tolist()
        
        substitution_map[token_id] = candidates

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(substitution_map, f, ensure_ascii=False, indent=2)
        
    print(f"🛡️ [DILIPLUS] Semantic Gatekeeper Ready! Substitution map saved to: {output_json}")

def main(settings=None):
    settings = settings or load_settings()
    project_root = settings.paths.root
    vocab_dir = settings.paths.vocab
    data_dir = settings.paths.data_cache
    os.makedirs(data_dir, exist_ok=True)
    
    vocab_poly_path = vocab_dir / 'vocab_polypharmacy.json'
    
    # 1. 生成模板 (向后兼容)
    build_polypharmacy_mapping_template(str(vocab_poly_path), str(settings.paths.drug_mapping_template))
    
    # 2. DILIPLUS 沙盒地图生成
    completed_csv = settings.paths.drug_mapping
    map_json = data_dir / 'Safety_Substitution_Map.json'
    generate_safety_substitution_map(str(completed_csv), str(map_json))

if __name__ == "__main__":
    main()

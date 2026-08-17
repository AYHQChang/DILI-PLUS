"""
DILI-PLUS | 动态事件与诊断词表构建（包实现）

职责：统计用药、化验项目和 ICD 编码，将药物与化验项目放入共享动态词表，
并为诊断建立独立词表；保留 PAD、UNK、CLS、SEP 特殊 Token。
输入：03_dili_dual_stream_tensors.parquet、03b_diag_tensors.parquet。
输出：vocab/vocab_polypharmacy.json、vocab/vocab_diagnosis.json。
状态：当前 DILI 单任务的数据编码步骤。
"""

import os
import pandas as pd
import numpy as np
import json
from collections import Counter
from diliplus.config import load_settings

def build_vocabulary(settings=None):
    # 所有相对路径均由配置对象锚定到项目根目录
    settings = settings or load_settings()
    data_dir = settings.model_data_dir
    vocab_dir = settings.paths.vocab
    os.makedirs(vocab_dir, exist_ok=True)
    
    med_tensor_path = data_dir / "03_dili_dual_stream_tensors.parquet"
    diag_tensor_path = data_dir / "03b_diag_tensors.parquet"
    
    # 基础 Transformer 特殊 Token
    base_vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    
    # ---------------------------------------------------------
    # 1. 构建临床动态事件词表 (Medications + Lab Items)
    # ---------------------------------------------------------
    print("⏳ [DILIPLUS] Building Dynamic Clinical Event Vocabulary...")
    if os.path.exists(med_tensor_path):
        df_med = pd.read_parquet(med_tensor_path)
        event_counter = Counter()
        
        # 统计药物
        for tokens in df_med['med_tokens'].dropna():
            if isinstance(tokens, (list, np.ndarray)):
                event_counter.update(tokens)
                
        # 统计化验项 (合并进入同一套动态词表，方便底层共用 Embedding)
        for tokens in df_med['lab_tokens'].dropna():
            if isinstance(tokens, (list, np.ndarray)):
                event_counter.update(tokens)
                
        med_vocab = base_vocab.copy()
        # 按出现频率排序分配 ID
        for word, count in sorted(
            event_counter.items(), key=lambda item: (-item[1], str(item[0]))
        ):
            med_vocab[word] = len(med_vocab)
            
        with open(vocab_dir / "vocab_polypharmacy.json", "w", encoding="utf-8") as f:
            json.dump(med_vocab, f, ensure_ascii=False, indent=2)
        print(f"   ✅ Dynamic Vocab Built: {len(med_vocab)} total tokens.")
    else:
        print(f"🚨 Missing Tensor file: {med_tensor_path}")

    # ---------------------------------------------------------
    # 2. 构建基线诊断词表 (Diagnosis)
    # ---------------------------------------------------------
    print("⏳ [DILIPLUS] Building Diagnostic Vocabulary...")
    if os.path.exists(diag_tensor_path):
        df_diag = pd.read_parquet(diag_tensor_path)
        diag_counter = Counter()
        
        for codes in df_diag['icd_codes'].dropna():
            if isinstance(codes, (list, np.ndarray)):
                diag_counter.update(codes)
                
        diag_vocab = base_vocab.copy()
        for word, count in sorted(
            diag_counter.items(), key=lambda item: (-item[1], str(item[0]))
        ):
            diag_vocab[word] = len(diag_vocab)
            
        with open(vocab_dir / "vocab_diagnosis.json", "w", encoding="utf-8") as f:
            json.dump(diag_vocab, f, ensure_ascii=False, indent=2)
        print(f"   ✅ Diagnosis Vocab Built: {len(diag_vocab)} total tokens.")
    else:
        print(f"🚨 Missing Diagnostic Tensor file: {diag_tensor_path}")
        
    print(f"🚀 [DILIPLUS] Tokenization Complete. System is now PyTorch-Ready!")

if __name__ == "__main__":
    build_vocabulary()

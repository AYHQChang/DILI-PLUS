"""
DILI-PLUS | PyTorch 数据集与张量对齐（包实现）

职责：合并动态序列与诊断 Parquet，执行 Token 编码、定长截断、填充和掩码构建。
输入：03_dili_dual_stream_tensors.parquet、03b_diag_tensors.parquet 及两套词表。
输出：DILIPlusDataset 单样本字典，包含 9 个模型输入张量和单一 label。
状态：当前 DILI 单任务训练、评估和解释脚本共用的数据入口。
"""
import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class DILIPlusDataset(Dataset):
    def __init__(self, data_dir, vocab_dir, max_med_len=100, max_lab_len=20, max_diag_len=15):
        # 1. 读取词表
        with open(os.path.join(vocab_dir, "vocab_polypharmacy.json"), 'r', encoding='utf-8') as f:
            self.med_vocab = json.load(f)
        with open(os.path.join(vocab_dir, "vocab_diagnosis.json"), 'r', encoding='utf-8') as f:
            self.diag_vocab = json.load(f)
            
        # 2. 读取合并好的 03 数据
        print("⏳ [DataLoader] Loading and merging dual-stream parquet files...")
        df_med_lab = pd.read_parquet(os.path.join(data_dir, "03_dili_dual_stream_tensors.parquet"))
        df_diag = pd.read_parquet(os.path.join(data_dir, "03b_diag_tensors.parquet"))
        
        # 使用 Left Join 保证队列完整性
        self.data = pd.merge(df_med_lab, df_diag, on='encounter_id', how='left')
        self.labels = self.data['label_dili'].values
        
        # 预设截断长度 (可根据 VRAM 调整)
        self.max_med_len = max_med_len
        self.max_lab_len = max_lab_len
        self.max_diag_len = max_diag_len

    def __len__(self):
        return len(self.labels)
        
    def _encode_tokens(self, tokens, vocab, max_len):
        if not isinstance(tokens, (list, np.ndarray)) or len(tokens) == 0:
            return [0], 1 # 空列表给个 [PAD]
        seq = [vocab.get(t, vocab.get('[UNK]', 1)) for t in tokens][:max_len]
        return seq, len(seq)
        
    def _pad_seq(self, seq, max_len, pad_value=0):
        if len(seq) >= max_len: 
            return seq[:max_len]
        return seq + [pad_value] * (max_len - len(seq))

    def _safe_list(self, val):
        """防止 Parquet 中的 NaN 导致类型错误"""
        if isinstance(val, (list, np.ndarray)):
            return list(val)
        return []

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # ---------------------------------------------------------
        # Stream A: Med 流
        # ---------------------------------------------------------
        med_seq, med_len = self._encode_tokens(row.get('med_tokens', []), self.med_vocab, self.max_med_len)
        med_dt = self._safe_list(row.get('med_dt_hours', []))
        dt_med = med_dt[:self.max_med_len] if len(med_dt) > 0 else [0.0]
        
        # ---------------------------------------------------------
        # Stream B: Lab 流 (异构三元组)
        # ---------------------------------------------------------
        lab_seq, lab_len = self._encode_tokens(row.get('lab_tokens', []), self.med_vocab, self.max_lab_len)
        lab_v = self._safe_list(row.get('lab_values', []))
        v_lab = lab_v[:self.max_lab_len] if len(lab_v) > 0 else [0.0]
        lab_dt = self._safe_list(row.get('lab_dt_hours', []))
        dt_lab = lab_dt[:self.max_lab_len] if len(lab_dt) > 0 else [0.0]
        
        # ---------------------------------------------------------
        # Stream C: Diag 流
        # ---------------------------------------------------------
        diag_seq, diag_len = self._encode_tokens(row.get('icd_codes', []), self.diag_vocab, self.max_diag_len)
        
        # 拼装返回字典
        return {
            'x_med': torch.tensor(self._pad_seq(med_seq, self.max_med_len, 0), dtype=torch.long),
            'dt_med': torch.tensor(self._pad_seq(dt_med, self.max_med_len, 0.0), dtype=torch.float),
            'mask_med': torch.tensor(self._pad_seq([1]*med_len, self.max_med_len, 0), dtype=torch.long),
            
            'x_lab': torch.tensor(self._pad_seq(lab_seq, self.max_lab_len, 0), dtype=torch.long),
            'v_lab': torch.tensor(self._pad_seq(v_lab, self.max_lab_len, 0.0), dtype=torch.float),
            'dt_lab': torch.tensor(self._pad_seq(dt_lab, self.max_lab_len, 0.0), dtype=torch.float),
            'mask_lab': torch.tensor(self._pad_seq([1]*lab_len, self.max_lab_len, 0), dtype=torch.long),
            
            'x_diag': torch.tensor(self._pad_seq(diag_seq, self.max_diag_len, 0), dtype=torch.long),
            'mask_diag': torch.tensor(self._pad_seq([1]*diag_len, self.max_diag_len, 0), dtype=torch.long),
            
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_vocab_sizes(vocab_dir):
    """读取词表维度，喂给模型初始化"""
    with open(os.path.join(vocab_dir, "vocab_polypharmacy.json"), 'r', encoding='utf-8') as f:
        v_med = len(json.load(f))
    with open(os.path.join(vocab_dir, "vocab_diagnosis.json"), 'r', encoding='utf-8') as f:
        v_diag = len(json.load(f))
    return {"vocab_med_size": v_med, "vocab_lab_size": v_med, "vocab_diag_size": v_diag}

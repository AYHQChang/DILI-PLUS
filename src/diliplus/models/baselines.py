"""
DILI-PLUS | 统一输入接口的深度学习对照模型（包实现）

职责：提供 MultiModalTextCNN、MultiModalBiLSTM 和从零训练的
MultimodalTransformerBaseline，使其接收与主模型相同的用药、化验和诊断输入。
输出：单一 AHI-proxy 二分类 logits。
状态：当前模型比较实验使用的对照架构。
维护说明：下方 imports 之前保留的是注释化 V2 参考实现，不参与 Python 执行；
实际生效的是 V3 类定义，后续目录重构时可迁入 archive/。
"""

# 以下为注释化的 V2 参考实现，仅用于历史追溯。
# """
# =============================================================================
# Module: models/baseline_models.py (DILI-PLUS Fair Comparison Edition V2)
# Phase: Stage 7 - Baseline Architectures
# Purpose: 
#     定义用于论文对比实验的经典序列模型，提供绝对公平的 3 流输入对照组。
#     包含: MultiModalTextCNN, MultiModalBiLSTM, MultiModalBaselineMedBERT。
#     [完整保留]: 原版模型所有的 Masked Pooling 逻辑与 LSTM 物理长度防崩溃约束。
#     [核心重构]: 对齐 x_med, dt_med, mask_med, x_lab, v_lab, dt_lab, mask_lab, x_diag, mask_diag。
#     [消融设计]: 不使用 Time2Vec，不使用参数化指数衰减，以此反衬 DILIPLUS。
# =============================================================================
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# # -----------------------------------------------------------------------------
# # 共享组件：基础位置编码 & 静态编码器 (保留原版逻辑)
# # -----------------------------------------------------------------------------
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, max_len: int = 5000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0))

#     def forward(self, x):
#         return x + self.pe[:, :x.size(1), :]

# class StaticProfileEncoder(nn.Module):
#     def __init__(self, diag_vocab_size: int, hidden_size: int = 128, dropout: float = 0.3):
#         super().__init__()
#         self.diag_embedding = nn.Embedding(diag_vocab_size, hidden_size, padding_idx=0)
#         self.projection = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x_diag, diag_mask):
#         emb_diag = self.diag_embedding(x_diag)
#         mask_expanded = diag_mask.unsqueeze(-1).float()
#         sum_x = torch.sum(emb_diag * mask_expanded, dim=1)
#         sum_mask = torch.clamp(torch.sum(mask_expanded, dim=1), min=1e-9)
#         out = sum_x / sum_mask
#         return self.projection(out)

# # -----------------------------------------------------------------------------
# # 传统基线 1: MultiModal BiLSTM (保留防崩溃强约束解包)
# # -----------------------------------------------------------------------------
# class MultiModalBiLSTM(nn.Module):
#     def __init__(self, vocab_med_size, vocab_lab_size, vocab_diag_size, hidden_size=128, dropout=0.3):
#         super().__init__()
#         # Stream A (Med)
#         self.med_emb = nn.Embedding(vocab_med_size, hidden_size, padding_idx=0)
#         self.med_lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=2, batch_first=True, bidirectional=True)
        
#         # Stream B (Lab)
#         self.lab_item_emb = nn.Embedding(vocab_lab_size, hidden_size, padding_idx=0)
#         self.lab_val_proj = nn.Linear(1, hidden_size)
#         self.lab_lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        
#         # Stream C (Diag)
#         self.static_encoder = StaticProfileEncoder(vocab_diag_size, hidden_size, dropout)
        
#         self.fusion_projection = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.ReLU(), nn.Dropout(dropout))
#         self.dili_head = nn.Linear(hidden_size, 2)

#     def forward(self, x_med, dt_med, mask_med, x_lab, v_lab, dt_lab, mask_lab, x_diag, mask_diag):
#         # --- Med 流处理 (完全继承您的原版掩码逻辑) ---
#         emb_m = self.med_emb(x_med)
#         lengths_m = torch.clamp(mask_med.sum(1), min=1).cpu()
#         packed_m = nn.utils.rnn.pack_padded_sequence(emb_m, lengths_m, batch_first=True, enforce_sorted=False)
#         packed_out_m, _ = self.med_lstm(packed_m)
#         out_m_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_out_m, batch_first=True, total_length=x_med.size(1))
        
#         mask_m_exp = mask_med.unsqueeze(-1).float()
#         h_med = torch.sum(out_m_seq * mask_m_exp, dim=1) / torch.clamp(mask_m_exp.sum(1), min=1e-9)

#         # --- Lab 流处理 ---
#         emb_l = self.lab_item_emb(x_lab) + self.lab_val_proj(v_lab.unsqueeze(-1))
#         lengths_l = torch.clamp(mask_lab.sum(1), min=1).cpu()
#         packed_l = nn.utils.rnn.pack_padded_sequence(emb_l, lengths_l, batch_first=True, enforce_sorted=False)
#         packed_out_l, _ = self.lab_lstm(packed_l)
#         out_l_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_out_l, batch_first=True, total_length=x_lab.size(1))
        
#         mask_l_exp = mask_lab.unsqueeze(-1).float()
#         h_lab = torch.sum(out_l_seq * mask_l_exp, dim=1) / torch.clamp(mask_l_exp.sum(1), min=1e-9)

#         # --- Diag 流处理 ---
#         h_diag = self.static_encoder(x_diag, mask_diag)
        
#         h_fused = self.fusion_projection(torch.cat([h_med, h_lab, h_diag], dim=-1))
#         return {"logits": self.dili_head(h_fused)}

# # -----------------------------------------------------------------------------
# # 传统基线 2: MultiModal TextCNN (完全继承您的原版通道拼接逻辑)
# # -----------------------------------------------------------------------------
# class MultiModalTextCNN(nn.Module):
#     def __init__(self, vocab_med_size, vocab_lab_size, vocab_diag_size, hidden_size=128, dropout=0.3):
#         super().__init__()
#         self.med_emb = nn.Embedding(vocab_med_size, hidden_size, padding_idx=0)
#         self.convs_med = nn.ModuleList([nn.Conv1d(hidden_size, hidden_size//3, k) for k in [2, 3, 4]])
        
#         self.lab_item_emb = nn.Embedding(vocab_lab_size, hidden_size, padding_idx=0)
#         self.lab_val_proj = nn.Linear(1, hidden_size)
#         self.convs_lab = nn.ModuleList([nn.Conv1d(hidden_size, hidden_size//2, k) for k in [2, 3]])
        
#         self.static_encoder = StaticProfileEncoder(vocab_diag_size, hidden_size, dropout)
        
#         # 为了保证 concat 后维度仍为 3*hidden_size (Med+Lab+Diag)
#         self.fusion_projection = nn.Sequential(
#             nn.Linear((hidden_size//3)*3 + (hidden_size//2)*2 + hidden_size, hidden_size), 
#             nn.ReLU(), nn.Dropout(dropout)
#         )
#         self.dili_head = nn.Linear(hidden_size, 2)

#     def forward(self, x_med, dt_med, mask_med, x_lab, v_lab, dt_lab, mask_lab, x_diag, mask_diag):
#         emb_m = self.med_emb(x_med).permute(0, 2, 1)
#         conv_out_m = [F.relu(conv(emb_m)) for conv in self.convs_med]
#         h_med = torch.cat([F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_out_m], 1)
        
#         emb_l = (self.lab_item_emb(x_lab) + self.lab_val_proj(v_lab.unsqueeze(-1))).permute(0, 2, 1)
#         pool_out_l = []
#         for conv in self.convs_lab:
#             if emb_l.size(2) >= conv.kernel_size[0]:
#                 out = F.relu(conv(emb_l))
#                 pool_out_l.append(F.max_pool1d(out, out.size(2)).squeeze(2))
#             else:
#                 pool_out_l.append(torch.zeros(emb_l.size(0), conv.out_channels, device=emb_l.device))
#         h_lab = torch.cat(pool_out_l, 1)
        
#         h_diag = self.static_encoder(x_diag, mask_diag)
        
#         h_fused = self.fusion_projection(torch.cat([h_med, h_lab, h_diag], dim=-1))
#         return {"logits": self.dili_head(h_fused)}

# # -----------------------------------------------------------------------------
# # 传统基线 3: MultiModal Baseline MedBERT (缺失连续时间编码的最强假想敌)
# # -----------------------------------------------------------------------------
# class MultiModalBaselineMedBERT(nn.Module):
#     def __init__(self, vocab_med_size, vocab_lab_size, vocab_diag_size, hidden_size=128, num_layers=2, num_heads=4, dropout=0.3):
#         super().__init__()
#         self.hidden_size = hidden_size
        
#         # Stream A: 传统的绝对位置编码 + Transformer
#         self.med_emb = nn.Embedding(vocab_med_size, hidden_size, padding_idx=0)
#         self.pos_encoder_med = PositionalEncoding(hidden_size)
#         layer_m = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True)
#         self.transformer_med = nn.TransformerEncoder(layer_m, num_layers=num_layers)
        
#         # Stream B: 传统的绝对位置编码 + Transformer (没有指数衰减！)
#         self.lab_item_emb = nn.Embedding(vocab_lab_size, hidden_size, padding_idx=0)
#         self.lab_val_proj = nn.Linear(1, hidden_size)
#         self.pos_encoder_lab = PositionalEncoding(hidden_size)
#         layer_l = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True)
#         self.transformer_lab = nn.TransformerEncoder(layer_l, num_layers=1)
        
#         self.static_encoder = StaticProfileEncoder(vocab_diag_size, hidden_size, dropout)
#         self.fusion_projection = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.ReLU(), nn.Dropout(dropout))
#         self.dili_head = nn.Linear(hidden_size, 2)

#     def forward(self, x_med, dt_med, mask_med, x_lab, v_lab, dt_lab, mask_lab, x_diag, mask_diag):
#         # Med Transformer
#         emb_m = self.med_emb(x_med) * math.sqrt(self.hidden_size)
#         emb_m = self.pos_encoder_med(emb_m)
#         key_pad_mask_m = ~mask_med.bool()
        
#         # 🔥 FIX 1: PyTorch Transformer NaN 防御！如果整行全为空，强行解开第0个Token的掩码
#         key_pad_mask_m[key_pad_mask_m.all(dim=1), 0] = False
        
#         out_m_seq = self.transformer_med(emb_m, src_key_padding_mask=key_pad_mask_m)
#         mask_m_exp = mask_med.unsqueeze(-1).float()
#         h_med = torch.sum(out_m_seq * mask_m_exp, dim=1) / torch.clamp(mask_m_exp.sum(1), min=1e-9)

#         # Lab Transformer
#         emb_l = (self.lab_item_emb(x_lab) + self.lab_val_proj(v_lab.unsqueeze(-1))) * math.sqrt(self.hidden_size)
#         emb_l = self.pos_encoder_lab(emb_l)
#         key_pad_mask_l = ~mask_lab.bool()
        
#         # 🔥 FIX 2: 同理，防御化验记录为空的患者
#         key_pad_mask_l[key_pad_mask_l.all(dim=1), 0] = False
        
#         out_l_seq = self.transformer_lab(emb_l, src_key_padding_mask=key_pad_mask_l)
#         mask_l_exp = mask_lab.unsqueeze(-1).float()
#         h_lab = torch.sum(out_l_seq * mask_l_exp, dim=1) / torch.clamp(mask_l_exp.sum(1), min=1e-9)

#         # Diag (保持不变)
#         h_diag = self.static_encoder(x_diag, mask_diag)
#         h_fused = self.fusion_projection(torch.cat([h_med, h_lab, h_diag], dim=-1))
#         return {"logits": self.dili_head(h_fused)}


# 以下为当前生效的 V3 实现；使用 torch.nan_to_num 处理缺失值传播。

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class StaticProfileEncoder(nn.Module):
    def __init__(self, diag_vocab_size: int, hidden_size: int = 128, dropout: float = 0.3):
        super().__init__()
        self.diag_embedding = nn.Embedding(diag_vocab_size, hidden_size, padding_idx=0)
        self.projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x_diag, diag_mask):
        emb_diag = self.diag_embedding(x_diag)
        emb_diag = torch.nan_to_num(emb_diag, nan=0.0) # 🔥 净化
        mask_expanded = diag_mask.unsqueeze(-1).float()
        sum_x = torch.sum(emb_diag * mask_expanded, dim=1)
        sum_mask = torch.clamp(torch.sum(mask_expanded, dim=1), min=1e-9)
        return self.projection(sum_x / sum_mask)

class MultiModalBiLSTM(nn.Module):
    def __init__(self, vocab_med_size, vocab_lab_size, vocab_diag_size, hidden_size=128, dropout=0.3):
        super().__init__()
        self.med_emb = nn.Embedding(vocab_med_size, hidden_size, padding_idx=0)
        self.med_lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=2, batch_first=True, bidirectional=True)
        self.lab_item_emb = nn.Embedding(vocab_lab_size, hidden_size, padding_idx=0)
        self.lab_val_proj = nn.Linear(1, hidden_size)
        self.lab_lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.static_encoder = StaticProfileEncoder(vocab_diag_size, hidden_size, dropout)
        self.fusion_projection = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.ReLU(), nn.Dropout(dropout))
        self.ahi_proxy_head = nn.Linear(hidden_size, 2)

    def forward(self, x_med, dt_med, mask_med, x_lab, v_lab, dt_lab, mask_lab, x_diag, mask_diag):
        v_lab = torch.nan_to_num(v_lab, nan=0.0) # 🔥 净化
        
        emb_m = self.med_emb(x_med)
        lengths_m = torch.clamp(mask_med.sum(1), min=1).cpu()
        packed_m = nn.utils.rnn.pack_padded_sequence(emb_m, lengths_m, batch_first=True, enforce_sorted=False)
        packed_out_m, _ = self.med_lstm(packed_m)
        out_m_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_out_m, batch_first=True, total_length=x_med.size(1))
        out_m_seq = torch.nan_to_num(out_m_seq, nan=0.0) # 🔥 净化
        mask_m_exp = mask_med.unsqueeze(-1).float()
        h_med = torch.sum(out_m_seq * mask_m_exp, dim=1) / torch.clamp(mask_m_exp.sum(1), min=1e-9)

        v_lab_norm = torch.log1p(torch.clamp(v_lab, min=0))
        emb_l = self.lab_item_emb(x_lab) + self.lab_val_proj(v_lab_norm.unsqueeze(-1))
        lengths_l = torch.clamp(mask_lab.sum(1), min=1).cpu()
        packed_l = nn.utils.rnn.pack_padded_sequence(emb_l, lengths_l, batch_first=True, enforce_sorted=False)
        packed_out_l, _ = self.lab_lstm(packed_l)
        out_l_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_out_l, batch_first=True, total_length=x_lab.size(1))
        out_l_seq = torch.nan_to_num(out_l_seq, nan=0.0) # 🔥 净化
        mask_l_exp = mask_lab.unsqueeze(-1).float()
        h_lab = torch.sum(out_l_seq * mask_l_exp, dim=1) / torch.clamp(mask_l_exp.sum(1), min=1e-9)

        h_diag = self.static_encoder(x_diag, mask_diag)
        h_fused = self.fusion_projection(torch.cat([h_med, h_lab, h_diag], dim=-1))
        return {"logits": self.ahi_proxy_head(h_fused)}

class MultiModalTextCNN(nn.Module):
    def __init__(self, vocab_med_size, vocab_lab_size, vocab_diag_size, hidden_size=128, dropout=0.3):
        super().__init__()
        self.med_emb = nn.Embedding(vocab_med_size, hidden_size, padding_idx=0)
        self.convs_med = nn.ModuleList([nn.Conv1d(hidden_size, hidden_size//3, k) for k in [2, 3, 4]])
        self.lab_item_emb = nn.Embedding(vocab_lab_size, hidden_size, padding_idx=0)
        self.lab_val_proj = nn.Linear(1, hidden_size)
        self.convs_lab = nn.ModuleList([nn.Conv1d(hidden_size, hidden_size//2, k) for k in [2, 3]])
        self.static_encoder = StaticProfileEncoder(vocab_diag_size, hidden_size, dropout)
        self.fusion_projection = nn.Sequential(nn.Linear((hidden_size//3)*3 + (hidden_size//2)*2 + hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout))
        self.ahi_proxy_head = nn.Linear(hidden_size, 2)

    @staticmethod
    def _masked_conv_max(sequence, mask, conv):
        """Pool only convolution windows whose every token is observable."""
        kernel = conv.kernel_size[0]
        if sequence.size(1) < kernel:
            return torch.zeros(
                sequence.size(0), conv.out_channels, device=sequence.device
            )
        masked = sequence * mask.unsqueeze(-1).to(sequence.dtype)
        features = F.relu(conv(masked.permute(0, 2, 1)))
        valid_windows = mask.bool().unfold(1, kernel, 1).all(dim=-1)
        features = features.masked_fill(~valid_windows.unsqueeze(1), float("-inf"))
        pooled = features.max(dim=2).values
        any_valid = valid_windows.any(dim=1, keepdim=True)
        return torch.where(any_valid, pooled, torch.zeros_like(pooled))

    def forward(self, x_med, dt_med, mask_med, x_lab, v_lab, dt_lab, mask_lab, x_diag, mask_diag):
        v_lab = torch.nan_to_num(v_lab, nan=0.0) # 🔥 净化
        
        emb_m = self.med_emb(x_med)
        h_med = torch.cat(
            [self._masked_conv_max(emb_m, mask_med, conv) for conv in self.convs_med],
            dim=1,
        )
        
        v_lab_norm = torch.log1p(torch.clamp(v_lab, min=0))
        emb_l = self.lab_item_emb(x_lab) + self.lab_val_proj(v_lab_norm.unsqueeze(-1))
        h_lab = torch.cat(
            [self._masked_conv_max(emb_l, mask_lab, conv) for conv in self.convs_lab],
            dim=1,
        )
        
        h_diag = self.static_encoder(x_diag, mask_diag)
        h_fused = self.fusion_projection(torch.cat([h_med, h_lab, h_diag], dim=-1))
        return {"logits": self.ahi_proxy_head(h_fused)}

class MultimodalTransformerBaseline(nn.Module):
    """From-scratch multimodal Transformer baseline without continuous time encoding."""
    def __init__(self, vocab_med_size, vocab_lab_size, vocab_diag_size, hidden_size=128, num_layers=2, num_heads=4, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.med_emb = nn.Embedding(vocab_med_size, hidden_size, padding_idx=0)
        self.pos_encoder_med = PositionalEncoding(hidden_size)
        layer_m = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_med = nn.TransformerEncoder(layer_m, num_layers=num_layers)
        
        self.lab_item_emb = nn.Embedding(vocab_lab_size, hidden_size, padding_idx=0)
        self.lab_val_proj = nn.Linear(1, hidden_size)
        self.pos_encoder_lab = PositionalEncoding(hidden_size)
        layer_l = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_lab = nn.TransformerEncoder(layer_l, num_layers=1)
        
        self.static_encoder = StaticProfileEncoder(vocab_diag_size, hidden_size, dropout)
        self.fusion_projection = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.ReLU(), nn.Dropout(dropout))
        self.ahi_proxy_head = nn.Linear(hidden_size, 2)

    def forward(self, x_med, dt_med, mask_med, x_lab, v_lab, dt_lab, mask_lab, x_diag, mask_diag):
        v_lab = torch.nan_to_num(v_lab, nan=0.0) # 🔥 净化
        
        emb_m = self.med_emb(x_med) * math.sqrt(self.hidden_size)
        emb_m = self.pos_encoder_med(emb_m)
        key_pad_mask_m = ~mask_med.bool()
        key_pad_mask_m[key_pad_mask_m.all(dim=1), 0] = False
        
        seq_len_med = x_med.size(1)
        causal_mask_med = torch.triu(
            torch.ones(
                seq_len_med,
                seq_len_med,
                dtype=torch.bool,
                device=x_med.device,
            ),
            diagonal=1,
        )
        
        out_m_seq = self.transformer_med(emb_m, mask=causal_mask_med, src_key_padding_mask=key_pad_mask_m)
        out_m_seq = torch.nan_to_num(out_m_seq, nan=0.0) # 🔥 净化
        mask_m_exp = mask_med.unsqueeze(-1).float()
        h_med = torch.sum(out_m_seq * mask_m_exp, dim=1) / torch.clamp(mask_m_exp.sum(1), min=1e-9)

        v_lab_norm = torch.log1p(torch.clamp(v_lab, min=0))
        emb_l = (self.lab_item_emb(x_lab) + self.lab_val_proj(v_lab_norm.unsqueeze(-1))) * math.sqrt(self.hidden_size)
        emb_l = self.pos_encoder_lab(emb_l)
        key_pad_mask_l = ~mask_lab.bool()
        key_pad_mask_l[key_pad_mask_l.all(dim=1), 0] = False
        
        out_l_seq = self.transformer_lab(emb_l, src_key_padding_mask=key_pad_mask_l)
        out_l_seq = torch.nan_to_num(out_l_seq, nan=0.0) # 🔥 净化
        mask_l_exp = mask_lab.unsqueeze(-1).float()
        h_lab = torch.sum(out_l_seq * mask_l_exp, dim=1) / torch.clamp(mask_l_exp.sum(1), min=1e-9)

        h_diag = self.static_encoder(x_diag, mask_diag)
        h_fused = self.fusion_projection(torch.cat([h_med, h_lab, h_diag], dim=-1))
        return {"logits": self.ahi_proxy_head(h_fused)}


# Historical import compatibility only; this class has never loaded Med-BERT
# pretraining weights. Formal experiments use ``MultimodalTransformerBaseline``.
MultiModalBaselineMedBERT = MultimodalTransformerBaseline

"""
DILI-PLUS | 从零训练的时间感知三模态 Transformer（包实现）

职责：编码用药 Token/时间差、化验项目/数值/时间差和静态诊断三个模态，
通过 Transformer、掩码池化与门控融合输出单一 AHI-proxy 二分类 logits。
输入：DILIPlusDataset 提供的九个特征张量。
输出：logits 以及三个模态的聚合表示，供训练和局部敏感性分析使用。
状态：当前 AHI-proxy 单任务的核心深度模型；没有语言模型预训练或外部 Med-BERT 权重。
解释边界：时间编码用于表示事件间隔，不等同于真实药代动力学方程；模型输出为
观察性预测，不直接提供药物因果效应。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# [核心组件 1] 连续时间编码器 (Log-Uniform Time2Vec)
# -----------------------------------------------------------------------------
class LogUniformTime2Vec(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # 对数均匀初始化提供不同初始频率；其临床时间尺度含义需由实验验证。
        log_freqs = torch.empty(d_model).uniform_(-4, 4)
        self.omega = nn.Parameter(torch.exp(log_freqs)) 
        self.phi = nn.Parameter(torch.zeros(d_model))
        self.linear = nn.Linear(1, d_model)

    def forward(self, dt):
        dt_expanded = dt.unsqueeze(-1) 
        v_periodic = torch.sin(dt_expanded * self.omega + self.phi) 
        v_linear = self.linear(dt_expanded)                         
        return v_periodic + v_linear

# -----------------------------------------------------------------------------
# [核心组件 2] 掩码全局平均池化 (防爆装甲版)
# -----------------------------------------------------------------------------
class MaskedGAP(nn.Module):
    def forward(self, x, mask):
        # 将非有限值归零，避免其在掩码池化中继续传播
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) 
        mask_expanded = mask.unsqueeze(-1).float()
        sum_x = torch.sum(x * mask_expanded, dim=1) 
        sum_mask = torch.clamp(torch.sum(mask_expanded, dim=1), min=1e-9) 
        return sum_x / sum_mask

# -----------------------------------------------------------------------------
# TimeAwareMultimodalTransformer：当前时间感知三模态实现
# -----------------------------------------------------------------------------
class TimeAwareMultimodalTransformer(nn.Module):
    """Medication/laboratory Transformer streams plus a diagnosis embedding stream."""

    def __init__(
        self,
        vocab_med_size,
        vocab_lab_size,
        vocab_diag_size,
        hidden_size=128,
        num_heads=4,
        dropout=0.3,
        diagnosis_modality_dropout_prob=0.15,
    ):
        super().__init__()
        if not 0.0 <= diagnosis_modality_dropout_prob < 1.0:
            raise ValueError("diagnosis_modality_dropout_prob must be in [0, 1)")
        self.hidden_size = hidden_size
        self.diagnosis_modality_dropout_prob = float(
            diagnosis_modality_dropout_prob
        )
        
        # === Stream A: medication-event sequence ===
        self.med_embedding = nn.Embedding(vocab_med_size, hidden_size, padding_idx=0)
        self.med_time2vec = LogUniformTime2Vec(hidden_size)
        encoder_layer_m = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True)
        self.med_transformer = nn.TransformerEncoder(encoder_layer_m, num_layers=2)
        self.med_pool = MaskedGAP()
        self.ln_med = nn.LayerNorm(hidden_size)
        
        # === Stream B: laboratory-event sequence ===
        self.lab_item_embedding = nn.Embedding(vocab_lab_size, hidden_size, padding_idx=0)
        self.lab_value_proj = nn.Linear(1, hidden_size)
        self.lab_time2vec = LogUniformTime2Vec(hidden_size)
        
        # Laboratory events are contextualized with a Transformer encoder.
        encoder_layer_l = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True)
        self.lab_transformer = nn.TransformerEncoder(encoder_layer_l, num_layers=2)
        self.lab_pool = MaskedGAP()
        self.ln_lab = nn.LayerNorm(hidden_size)
        
        # === Stream C: Static Phenotype (静态诊断流) ===
        self.diag_embedding = nn.Embedding(vocab_diag_size, hidden_size, padding_idx=0)
        self.diag_pool = MaskedGAP()
        self.ln_diag = nn.LayerNorm(hidden_size)
        
        # === 融合层 (残差门控) ===
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size * 2), nn.Sigmoid())
        self.fusion_projection = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.ReLU(), nn.Dropout(dropout))
        self.ahi_proxy_head = nn.Linear(hidden_size, 2)
        
    def forward(self, x_med, dt_med, mask_med, x_lab, v_lab, dt_lab, mask_lab, x_diag, mask_diag):
        
        # ---------------------------------------------------------
        # 1. 全局数据净化 & 物理连续变量对数平滑 (防爆装甲)
        # ---------------------------------------------------------
        v_lab = torch.nan_to_num(v_lab, nan=0.0)
        dt_med = torch.nan_to_num(dt_med, nan=0.0)
        dt_lab = torch.nan_to_num(dt_lab, nan=0.0)
        
        dt_med_norm = torch.log1p(torch.clamp(dt_med, min=0))
        dt_lab_norm = torch.log1p(torch.clamp(dt_lab, min=0))
        v_lab_norm = torch.log1p(torch.clamp(v_lab, min=0))
        
        # ---------------------------------------------------------
        # 2. Stream A 向前传播
        # ---------------------------------------------------------
        emb_med = self.med_embedding(x_med)             
        t_emb_med = self.med_time2vec(dt_med_norm)        
        h_med_seq = emb_med + t_emb_med
        
        med_key_pad_mask = ~mask_med.bool()
        med_key_pad_mask[med_key_pad_mask.all(dim=1), 0] = False # 防御全空掩码
        
        # 自回归上三角掩码，限制每个位置访问其后的用药事件
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
        
        h_med_seq = self.med_transformer(h_med_seq, mask=causal_mask_med, src_key_padding_mask=med_key_pad_mask)
        h_med = self.ln_med(self.med_pool(h_med_seq, mask_med))
        
        # ---------------------------------------------------------
        # 3. Laboratory stream
        # ---------------------------------------------------------
        emb_lab_item = self.lab_item_embedding(x_lab)   
        emb_lab_value = self.lab_value_proj(v_lab_norm.unsqueeze(-1)) 
        t_emb_lab = self.lab_time2vec(dt_lab_norm) 
        
        # Add the laboratory-item, value and learned time representations.
        h_lab_seq = emb_lab_item + emb_lab_value + t_emb_lab
        
        lab_key_pad_mask = ~mask_lab.bool()
        lab_key_pad_mask[lab_key_pad_mask.all(dim=1), 0] = False
        
        # Model interactions among the retained pre-prediction laboratory events.
        h_lab_seq = self.lab_transformer(h_lab_seq, src_key_padding_mask=lab_key_pad_mask)
        h_lab = self.ln_lab(self.lab_pool(h_lab_seq, mask_lab))
        
        # ---------------------------------------------------------
        # 4. Stream C 向前传播
        # ---------------------------------------------------------
        emb_diag = self.diag_embedding(x_diag)
        h_diag = self.ln_diag(self.diag_pool(emb_diag, mask_diag))
        
        # ---------------------------------------------------------
        # 5. Diagnosis-modality dropout（正则化，不声称保持表示期望）
        # ---------------------------------------------------------
        if self.training and self.diagnosis_modality_dropout_prob > 0:
            keep_diag = torch.rand(
                (h_diag.size(0), 1), device=h_diag.device
            ) >= self.diagnosis_modality_dropout_prob
            h_diag = h_diag * keep_diag.to(h_diag.dtype)
        
        # ---------------------------------------------------------
        # 6. 残差门控融合 (Residual Gating Fusion)
        # ---------------------------------------------------------
        h_concat = torch.cat([h_med, h_lab, h_diag], dim=-1) # [B, 3H]
        h_fused_raw = self.fusion_projection(h_concat)       # [B, H]
        gate = self.fusion_gate(h_concat)                    # [B, 2H]
        
        # 提取动态特征联合体
        h_dynamic = h_med + h_lab                            # [B, H]
        
        # 门控截流与动态残差混入
        h_fused = gate[:, :self.hidden_size] * h_dynamic + gate[:, self.hidden_size:] * h_fused_raw
        
        logits = self.ahi_proxy_head(h_fused)
        
        # 返回模态表示，供局部归因和输入扰动敏感性分析复用
        return {
            "logits": logits,
            "h_med": h_med,
            "h_lab": h_lab,
            "h_diag": h_diag
        }


# Historical import compatibility only. Formal runs use the truthful class and
# artifact name ``TimeAwareMultimodalTransformer`` from ``models.registry``.
DILIPlusEngine = TimeAwareMultimodalTransformer

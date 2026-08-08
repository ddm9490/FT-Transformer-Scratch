import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ffn=None, dropout_p=0.1): # 1. 드롭아웃 추가
        super().__init__()
        if d_ffn is None:
            d_ffn = int(2 * (4 * d_model / 3))
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_model, d_ffn, bias=False)
        self.w3 = nn.Linear(d_ffn, d_model, bias=False)
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x):
        hidden = F.silu(self.w1(x)) * self.w2(x)
        return self.w3(self.drop(hidden))

class GatedMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout_p = 0.2, gated_attn = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout_p
        self.gated_attn = gated_attn

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        if self.gated_attn:
            self.gate_proj = nn.Linear(self.head_dim, self.head_dim, bias=True)
            nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
            nn.init.constant_(self.gate_proj.bias, 2.0)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Any:
        batch_size, seq_len, _ = query.shape
        k_seq_len = key.shape[1]

        # Shape: (batch_size, num_heads, seq_len, head_dim)
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout_p if self.training else 0.0  # 🌟 원래대로 다시 0.1 복구!
        )  # Shape: (batch_size,`` num_heads, seq_len, head_dim)

        if self.gated_attn:
            gate = torch.sigmoid(self.gate_proj(q))  # Shape: (batch_size, num_heads, seq_len, head_dim)
            gated_attn_out = attn_out * gate
            out = gated_attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        else:
            out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(out)

class AttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout_p=0.1, ffn_module=None, is_self_attn=True, gated_attn = True, gated_AFR = True):
        super().__init__()
        self.is_self_attn = is_self_attn
        self.gated_AFR = gated_AFR

        self.attn = GatedMultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout_p=dropout_p, gated_attn = gated_attn)
        self.ffn = ffn_module if ffn_module is not None else SwiGLUFFN(d_model, dropout_p=dropout_p)

        self.attn_ln = nn.RMSNorm(d_model)
        self.ffn_ln = nn.RMSNorm(d_model)

        self.attn_drop = nn.Dropout(dropout_p)
        self.combined_drop = nn.Dropout(dropout_p)

        self.kv_ln = nn.Identity() if is_self_attn else nn.RMSNorm(d_model)

        if gated_AFR:
            self.res_gate_proj = nn.Linear(d_model, d_model, bias=True)
            nn.init.normal_(self.res_gate_proj.weight, mean=0.0, std=0.02)
            nn.init.constant_(self.res_gate_proj.bias, 1.0)

            
    def forward(self, query, key, value):
        ln_query = self.attn_ln(query)

        if self.is_self_attn:
            ln_key = ln_query
            ln_value = ln_query
        else:
            ln_key = self.kv_ln(key)
            ln_value = self.kv_ln(value)

        attn_out = self.attn(query=ln_query, key=ln_key, value=ln_value)
        attn_out_dropped = self.attn_drop(attn_out) 

        x = query + attn_out_dropped
        ffn_in = self.ffn_ln(x)
        ffn_out = self.ffn(ffn_in)
        
        if self.gated_AFR:
            res_gate = torch.sigmoid(self.res_gate_proj(ln_query)) 
            combined_out = res_gate * attn_out + (1.0 - res_gate) * ffn_out
        else:
            combined_out = ffn_out
        return x + self.combined_drop(combined_out)

class AttnRes(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_l = nn.Parameter(torch.zeros(1, d_model))

    def forward(self, history_list):
        # history_tensor : [B, Depth, T, D]
        history_tensor = torch.stack(history_list, dim=1)
        normed_history = F.normalize(history_tensor, p=2, dim=-1)
        
        d_k = self.w_l.shape[-1]
        scores = torch.einsum("bDtd, d -> bDt", normed_history, self.w_l.squeeze(0)) / (d_k ** 0.5)
        alpha = F.softmax(scores, dim=1).unsqueeze(-1)
        
        return torch.sum(alpha * history_tensor, dim=1)


class SelfAttentionNetAttnres(nn.Module):
    def __init__(self, d_model, num_layers, num_heads=4, dropout_p=0.1, share_ffn=False, gated_attn = True, gated_AFR = True):
        super().__init__()
        shared_ffn = SwiGLUFFN(d_model, dropout_p=dropout_p) if share_ffn else None

        self.blocks = nn.ModuleList([
            AttentionBlock(
                d_model, 
                num_heads, 
                dropout_p=dropout_p, 
                ffn_module=shared_ffn, 
                gated_attn = gated_attn, 
                gated_AFR = gated_AFR
                ) 
            for _ in range(num_layers)
        ])
        self.attnres_layers = nn.ModuleList([AttnRes(d_model) for _ in range(num_layers)])


    def forward(self, query):
        history = [query]
        for block, attnres_layer in zip(self.blocks, self.attnres_layers):
            Q_base = attnres_layer(history)
            current_query = block(query=Q_base, key=Q_base, value=Q_base)
            history.append(current_query)

        return history[-1]

class FTTModel(nn.Module):
    def __init__(self, out_dim, d_model, num_self_attn_layers=2, nhead=4, dropout_p=0.1, share_ffn=False, gated_attn = True, gated_AFR = True):
        super().__init__()

        self.d_model = d_model

        self.backbone_net = SelfAttentionNetAttnres(
            d_model=d_model,
            num_layers=num_self_attn_layers,
            num_heads=nhead,
            dropout_p=dropout_p,
            share_ffn=share_ffn,
            gated_attn = gated_attn,
            gated_AFR = gated_AFR
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x_num, x_cat, tokenizer):
        embedded_vector = tokenizer(x_num, x_cat) # [Batch, Total_features, d_model]
        B = embedded_vector.size(0)

        cls_query = self.cls_token.expand(B, -1, -1)
        query_concated = torch.cat([cls_query, embedded_vector], dim=1) # [Batch, Total_features + 1, d_model]

        self_attn_outputs = self.backbone_net(query=query_concated)
        
        cls_token_out = self_attn_outputs[:, 0] # [Batch, d_model]

        return self.head(cls_token_out)

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
        # 활성화 이후 드롭아웃을 걸어 뉴런 암기를 방지
        hidden = F.silu(self.w1(x)) * self.w2(x)
        return self.w3(self.drop(hidden))

class GatedMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout_p = 0.2):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout_p

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Q, K, V Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.1))

        # 🌟 Query 기반 Gate Projection (Head별로 적용하기 위해 head_dim 차원 사용)
        
        self.gate_proj = nn.Linear(self.head_dim, self.head_dim, bias=True)
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.gate_proj.bias, 2.0)
        
        # Final Output Projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Any:
        batch_size, seq_len, _ = query.shape
        k_seq_len = key.shape[1]

        # 1. Linear Projection & Multi-head reshape
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Scaled Dot-Product Attention (PyTorch 최적화 API 활용)
        # Fast Attention (FlashAttention 계열) 백엔드를 그대로 사용할 수 있어 효율적입니다.
        attn_out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout_p if self.training else 0.0  # 🌟 원래대로 다시 0.1 복구!
        )  # Shape: (batch_size, num_heads, seq_len, head_dim)

        # 3. 🌟 Query 기반 Post-Attention Output Gating
        # Head별 Query(q)를 Gate Projection에 넣고 Sigmoid 처리
        gate = torch.sigmoid(self.gate_proj(q))  # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Hadamard Product (원소별 곱)으로 문맥 정보 필터링
        gated_attn_out = attn_out * gate

        # 4. Concatenate Heads & Final Linear Projection
        # Shape: (batch_size, seq_len, d_model)
        out = gated_attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)

class AttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout_p=0.1, ffn_module=None, is_self_attn=True):
        super().__init__()
        self.is_self_attn = is_self_attn

        self.attn = GatedMultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout_p=dropout_p)
        self.ffn = ffn_module if ffn_module is not None else SwiGLUFFN(d_model, dropout_p=dropout_p)

        self.attn_ln = nn.RMSNorm(d_model)
        self.ffn_ln = nn.RMSNorm(d_model)

        self.attn_drop = nn.Dropout(dropout_p)
        self.ffn_drop = nn.Dropout(dropout_p)

        self.kv_ln = nn.Identity() if is_self_attn else nn.RMSNorm(d_model)


        self.res_gate_proj = nn.Linear(d_model, d_model, bias=True) 
        self.res_gate_drop = nn.Dropout(dropout_p)   
        nn.init.normal_(self.res_gate_proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.res_gate_proj.bias, 0.0)

    def forward(self, query, key, value):
        ln_query = self.attn_ln(query)

        if self.is_self_attn:
            ln_key = ln_query
            ln_value = ln_query
        else:
            ln_key = self.kv_ln(key)
            ln_value = self.kv_ln(value)

        attn_out = self.attn(query=ln_query, key=ln_key, value=ln_value)
        x = query + self.attn_drop(attn_out)

        ln_x = self.ffn_ln(x)
        ffn_out = self.ffn(ln_x)

        res_gate = torch.sigmoid(self.res_gate_proj(attn_out))
        res_gate = self.res_gate_drop(res_gate)
        mixed_ffn_out = res_gate * attn_out + (1.0 - res_gate) * ffn_out

        return x + self.ffn_drop(mixed_ffn_out)


class AttnRes(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_l = nn.Parameter(torch.zeros(1, d_model))

    def forward(self, history_list):
        # [B, Depth, T, D]
        history_tensor = torch.stack(history_list, dim=1)
        normed_history = F.normalize(history_tensor, p=2, dim=-1)
        
        d_k = self.w_l.shape[-1]
        # einsum으로 축 매핑 명확화
        scores = torch.einsum("bDtd, d -> bDt", normed_history, self.w_l.squeeze(0)) / (d_k ** 0.5)
        alpha = F.softmax(scores, dim=1).unsqueeze(-1)
        
        return torch.sum(alpha * history_tensor, dim=1)


class SelfAttentionNetAttnres(nn.Module):
    def __init__(self, d_model, num_layers, num_heads=4, dropout_p=0.1, share_ffn=False):
        super().__init__()
        shared_ffn = SwiGLUFFN(d_model, dropout_p=dropout_p) if share_ffn else None

        # 💡 ModuleList 정의를 콤팩트하게 변경
        self.blocks = nn.ModuleList([
            AttentionBlock(d_model, num_heads, dropout_p=dropout_p, ffn_module=shared_ffn) 
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
    def __init__(self, out_dim, d_model, num_self_attn_layers=2, nhead=4, dropout_p=0.1, share_ffn=False):
        super().__init__()

        self.d_model = d_model

        # 💡 FTTBlock의 역할을 FTTModel이 흡수하여 구조를 단순화
        self.backbone_net = SelfAttentionNetAttnres(
            d_model=d_model,
            num_layers=num_self_attn_layers,
            num_heads=nhead,
            dropout_p=dropout_p,
            share_ffn=share_ffn
        )
        
        # CLS 토큰 정의 및 초기화
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # 회귀용 최종 출력 레이어
        self.regressor = nn.Linear(d_model, out_dim)

    def forward(self, x_num, x_cat, tokenizer):
        # 1. 토크나이저를 통한 피처 인베딩
        embedded_vector = tokenizer(x_num, x_cat) # [Batch, Total_features, d_model]
        B = embedded_vector.size(0)

        # 2. CLS 토큰 결합
        cls_query = self.cls_token.expand(B, -1, -1)
        query_concated = torch.cat([cls_query, embedded_vector], dim=1) # [Batch, Total_features + 1, d_model]

        # 3. AttnRes 백본 네트워크 통과
        self_attn_outputs = self.backbone_net(query=query_concated)

        # 4. 💡 0번 인덱스의 CLS 토큰만 안전하게 추출 (squeeze 제거로 버그 방지)
        cls_token_out = self_attn_outputs[:, 0] # [Batch, d_model]

        # 5. 최종 예측값 반환
        return self.regressor(cls_token_out)

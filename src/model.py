import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ffn=None, dropout_p=0.1):
        super().__init__()
        if d_ffn is None:
            d_ffn = int(2 * (4 * d_model / 3))
        else:
            d_ffn = int(d_ffn)
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_model, d_ffn, bias=False)
        self.w3 = nn.Linear(d_ffn, d_model, bias=False)
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x):
        hidden = F.silu(self.w1(x)) * self.w2(x)
        return self.w3(self.drop(hidden))

class GatedMultiHeadAttention(nn.Module):
    def __init__(
        self, 
        d_query: int,          
        d_kv = None,
        num_heads: int = 4, 
        dropout_p: float = 0.2, 
        gated_attn: bool = False
    ):
        super().__init__()
        self.d_query = d_query
        self.d_kv = d_kv if d_kv is not None else d_query
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.gated_attn = gated_attn

        assert d_query % num_heads == 0, "d_query must be divisible by num_heads"
        self.head_dim = d_query // num_heads

        self.q_proj = nn.Linear(self.d_query, self.d_query, bias=False)
        self.k_proj = nn.Linear(self.d_kv, self.d_query, bias=False) 
        self.v_proj = nn.Linear(self.d_kv, self.d_query, bias=False) 
        
        if self.gated_attn:
            self.gate_proj = nn.Linear(self.head_dim, self.head_dim, bias=True)
            self._reset_parameters()

        self.out_proj = nn.Linear(self.d_query, self.d_query, bias=False)
    
    def _reset_parameters(self):
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.gate_proj.bias, 2.0)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Any:
        batch_size, q_seq_len, _ = query.shape
        kv_seq_len = key.shape[1]

        q = self.q_proj(query).view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout_p if self.training else 0.0 
        )  

        if self.gated_attn:
            gate = torch.sigmoid(self.gate_proj(q))
            gated_attn_out = attn_out * gate
            out = gated_attn_out.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.d_query)
        else:
            out = attn_out.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.d_query)

        return self.out_proj(out)

class AttentionBlock(nn.Module):
    def __init__(
        self, 
        d_query: int,      
        d_kv = None,     
        d_ffn = None, 
        ffn_module = None, 
        num_heads = 4, 
        dropout_p = 0.1, 
        is_self_attn = True, 
        gated_attn = False, 
        gated_AFR = True
    ):
        super().__init__()
        self.d_query = d_query
        self.d_kv = d_kv if d_kv is not None else d_query
        self.is_self_attn = is_self_attn
        self.gated_AFR = gated_AFR

        self.attn = GatedMultiHeadAttention(
            d_query=d_query, 
            d_kv=self.d_kv, 
            num_heads=num_heads, 
            dropout_p=dropout_p, 
            gated_attn=gated_attn
        )
        
        self.ffn = ffn_module if ffn_module else SwiGLUFFN(d_query, d_ffn=d_ffn, dropout_p=dropout_p) 

        self.attn_ln = nn.LayerNorm(d_query)
        self.ffn_ln = nn.LayerNorm(d_query)

        self.attn_drop = nn.Dropout(dropout_p)
        self.combined_drop = nn.Dropout(dropout_p)

        self.kv_ln = nn.Identity() if is_self_attn else nn.LayerNorm(self.d_kv)

        if gated_AFR:
            self.res_gate_proj = nn.Linear(d_query, d_query, bias=True)
            self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.res_gate_proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.res_gate_proj.bias, 1.5)

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
            res_gate = torch.sigmoid(self.res_gate_proj(ffn_in)) 
            combined_out = res_gate * attn_out + (1.0 - res_gate) * ffn_out
        else:
            combined_out = ffn_out
            
        return x + self.combined_drop(combined_out)


class AttnRes(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_l = nn.Parameter(torch.zeros(1, d_model))

    def forward(self, history_list):
        history_tensor = torch.stack(history_list, dim=1)
        normed_history = F.normalize(history_tensor, p=2, dim=-1)
        
        d_k = self.w_l.shape[-1]
        scores = torch.einsum("bDtd, d -> bDt", normed_history, self.w_l.squeeze(0)) / (d_k ** 0.5)
        alpha = F.softmax(scores, dim=1).unsqueeze(-1)
        
        return torch.sum(alpha * history_tensor, dim=1)


class FTTBackboneNet(nn.Module):
    def __init__(
            self, 
            in_dim, 
            d_model, 
            num_layers, 
            num_latent_tokens = 16,
            num_heads=4, 
            dropout_p=0.1, 
            gated_attn = False, 
            gated_AFR = True
        ):
        super().__init__()

        self.self_blocks = nn.ModuleList([
            AttentionBlock(
                d_query = d_model, 
                num_heads = num_heads, 
                dropout_p=dropout_p, 
                gated_attn = gated_attn, 
                gated_AFR = gated_AFR
                ) 
            for _ in range(num_layers)
        ])

        # self.intersample_blocks = nn.ModuleList([
        #     BottleneckIntersampleAttention( 
        #         d_model = d_model, 
        #         d_latent = d_model // 2,
        #         num_features=in_dim, 
        #         num_latent_tokens= num_latent_tokens, 
        #         num_heads=num_heads // 2,
        #         dropout_p = dropout_p, 
        #         gated_attn = gated_attn,
        #         gated_AFR = gated_AFR
        #         )
        #     for _ in range(num_layers)
        # ])
        self.attnres_layers = nn.ModuleList([AttnRes(d_model) for _ in range(num_layers)])


    def forward(self, query):
        history = [query]

        for self_block ,attnres_layer in zip(self.self_blocks, self.attnres_layers):
            Q_base = attnres_layer(history)
            self_attn_out = self_block(query=Q_base, key=Q_base, value=Q_base)
            # intersample_attn_out = inter_block(self_attn_out)
            history.append(self_attn_out)

        return history[-1]

class BottleneckIntersampleAttention(nn.Module):
    def __init__(self, d_model, num_features, d_latent = None, num_latent_tokens = 8, num_heads=2, dropout_p=0.1, gated_attn = False, gated_AFR = True):
        super().__init__()

        self.d_latent = d_model if d_latent is None else d_latent

        latent_ffn_module = SwiGLUFFN(self.d_latent, dropout_p = dropout_p)
        model_ffn_module = latent_ffn_module if d_latent is None else SwiGLUFFN(d_model, dropout_p = dropout_p)
        
        self.in_latent_block = AttentionBlock(
            d_query = self.d_latent,
            d_kv=d_model,
            ffn_module=latent_ffn_module,
            num_heads = num_heads,
            dropout_p = dropout_p,
            is_self_attn=False,
            gated_attn=gated_attn
        )
        self.self_latent_block = AttentionBlock(
            d_query = self.d_latent,
            ffn_module=latent_ffn_module,
            num_heads = num_heads,
            dropout_p = dropout_p,
            is_self_attn=True,
            gated_attn=gated_attn
        )
        self.out_latent_block = AttentionBlock(
            d_query = d_model,
            d_kv = self.d_latent,
            ffn_module=model_ffn_module,
            num_heads = num_heads,
            dropout_p = dropout_p,
            is_self_attn=False,
            gated_attn=gated_attn
        )

        self.latents = nn.Parameter(torch.empty(num_features, num_latent_tokens, self.d_latent))
        self.gamma = nn.Parameter(torch.zeros(1))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.latents)

    def forward(self, x):
        x_permuted = x.permute(1,0,2).contiguous()
        
        latents_compressed = self.in_latent_block(
            query = self.latents,
            key = x_permuted,
            value = x_permuted,
        )

        latents_attn = self.self_latent_block(
            query = latents_compressed,
            key = latents_compressed,
            value = latents_compressed,
        )

        intersample_out = self.out_latent_block(
            query = x_permuted,
            key = latents_attn,
            value = latents_attn,
        )

        intersample_out = intersample_out.permute(1,0,2).contiguous()
        out = x + self.gamma * intersample_out
        return out


class FTTModel(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, num_self_attn_layers=2, num_heads=4, dropout_p=0.1, gated_attn = False, gated_AFR = True):
        super().__init__()

        self.d_model = d_model

        self.backbone_net = FTTBackboneNet(
            in_dim = in_dim+1,
            d_model=d_model,
            num_layers=num_self_attn_layers,
            num_heads=num_heads,
            dropout_p=dropout_p,
            gated_attn = gated_attn,
            gated_AFR = gated_AFR
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x_num, x_cat, tokenizer):
        embedded_vector = tokenizer(x_num, x_cat) 
        B = embedded_vector.size(0)
        
        cls_query = self.cls_token.expand(B, -1, -1)
        query_concated = torch.cat([cls_query, embedded_vector], dim=1)

        self_attn_outputs = self.backbone_net(query=query_concated)
        
        cls_token_out = self_attn_outputs[:, 0]

        return self.head(cls_token_out)

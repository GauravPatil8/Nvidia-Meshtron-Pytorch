import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from rotary_embedding_torch import RotaryEmbedding

class Attention(nn.Module):
    """flash Sliding window attention"""
    def __init__(self, dim, num_heads, head_dim, window_size, rope_emb: RotaryEmbedding, dropout: float):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout_p = dropout
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.rope = rope_emb 
        self.q_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, dim, bias=False)

    def forward(self, q, k, v, mask = None):
        """
            x: Torch.tensor((batch, seq_len, dim))
            mask: (batch, seq_len) or (batch, 1, seq_len, seq_len)
        """
        is_self_attn = (q is k)

        b_q, l_q, _ = q.size()
        b_k, l_k, _ = k.size()
        b_v, l_v, _ = v.size()
        h = self.num_heads
        d = self.head_dim


        q = self.q_proj(q).view(b_q, l_q, h, d).transpose(1,2)
        k = self.k_proj(k).view(b_k, l_k, h, d).transpose(1,2)
        v = self.v_proj(v).view(b_v, l_v, h, d)

        #positional embedding
        q = self.rope.rotate_queries_or_keys(q)

        if is_self_attn:
            k = self.rope.rotate_queries_or_keys(k)

        q = q.transpose(1,2)
        k = k.transpose(1,2)

        if is_self_attn:
            causal = True
            window_size = (self.window_size - 1, 0)
        else:
            causal = False
            window_size = (-1,-1)

        out = flash_attn_func(
            q, k, v,
            window_size=window_size,
            dropout_p = self.dropout_p,
            causal=causal
        )

        out = out.reshape(b_q, l_q, self.inner_dim)
        out = self.o_proj(out)
        
        return out

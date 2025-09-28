import torch
import torch.nn as nn
from flash_attn import flash_attn_func


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=bias, dtype=torch.float16)
        self.kv_proj = nn.Linear(d_model, d_model * 2, bias=bias, dtype=torch.float16)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias, dtype=torch.float16)

    def forward(self, q, kv):
        q = q.to(dtype=torch.float16)
        kv = kv.to(dtype=torch.float16)
        q_batch_size, q_seq_len, _ = q.shape
        kv_batch_size, kv_seq_len, _ = kv.shape

        q = self.q_proj(q)
        kv = self.kv_proj(kv)

        q = q.reshape(q_batch_size, q_seq_len, self.num_heads, self.head_dim)
        kv = kv.reshape(kv_batch_size, kv_seq_len, 2, self.num_heads, self.head_dim)

        k, v = kv.unbind(dim=2)

        output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=None,
            causal=True,
        )

        output = output.reshape(q_batch_size, q_seq_len, self.d_model)
        output = self.out_proj(output)
        
        return output
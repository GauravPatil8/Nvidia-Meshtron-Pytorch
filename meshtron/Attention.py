import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from flash_attn.layers.rotary import RotaryEmbedding
# from meshtron.rope import precompute_theta_pos_freq, apply_rope

class Attention(nn.Module):
    """flash Sliding window attention"""
    def __init__(self, dim, num_heads, head_dim, window_size):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        hidden_dim = num_heads * self.head_dim

        self.rope = RotaryEmbedding(dim=self.head_dim, base=10000.0)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, q, k, v, mask = None):
        """
            x: Torch.tensor((batch, seq_len, dim))
            mask: (batch, seq_len) or (batch, 1, seq_len, seq_len)
        """
        
        b_q, l_q, _ = q.size()
        b_k, l_k, _ = k.size()
        b_v, l_v, _ = v.size()
        h = self.num_heads
        d = self.head_dim


        q = self.q_proj(q).view(b_q, l_q, h, d)
        k = self.k_proj(k).view(b_k, l_k, h, d)
        v = self.v_proj(v).view(b_v, l_v, h, d)

        #positional embedding
        # q, k = self.rope(q, k)

        out = flash_attn_func(
            q, k, v,
            window_size=(self.window_size - 1, 0),
            causal=True
        )
        
        out = out.reshape(b_q, l_q, self.dim)
        out = self.o_proj(out)
        
        return out
                


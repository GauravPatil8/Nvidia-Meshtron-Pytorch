import torch
import torch.nn as nn
import torch.nn.functional as F
from meshtron.rope import precompute_theta_pos_freq, apply_rope

class Attention(nn.Module):
    """Sliding window attention"""
    def __init__(self, dim, num_heads, head_dim, window_size):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        hidden_dim = num_heads * self.head_dim

        self.q_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, dim, bias=False)

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
        device = q.device
        q_freqs_complex = precompute_theta_pos_freq(self.head_dim, l_q, 10000.0)
        k_freqs_complex = precompute_theta_pos_freq(self.head_dim, l_k, 10000.0)

        q = self.q_proj(q).view(b_q, l_q, h, d)
        k = self.k_proj(k).view(b_k, l_k, h, d)
        
        # b l h d -> b h l d
        v = self.v_proj(v).view(b_v, l_v, h, d).transpose(1,2)

        #positional embedding
        q = apply_rope(q, q_freqs_complex, device).transpose(1,2) 
        k = apply_rope(k, k_freqs_complex, device).transpose(1,2)

        attn_out = torch.zeros_like(q)

        for i in range(l_q):
            #window range
            start = max(0, i-self.window_size)
            end = i+1

            q_i = q[: , :, i : i+1, :]
            k_win = k[:, :, start:end, :]
            v_win = v[:, :, start:end, :]

            attn_scores  = torch.matmul(q_i, k_win.transpose(-2, -1)).mul_(d ** -0.5)

            # if mask is not None:
            #     if mask.dim() == 2:
            #         local_mask = mask[:, start:end].unsqueeze(1).unsqueeze(2)  # (B,1,1,W)
            #     elif mask.dim() == 3:
            #         local_mask = mask[:, i:i+1, start:end].unsqueeze(1)  # (B,1,1,W)
            #     elif mask.dim() == 4:
            #         local_mask = mask[:, :, i:i+1, start:end]
            #     attn_scores = attn_scores.masked_fill_(local_mask == 0, float('-inf'))

            attn_prob = F.softmax(attn_scores, dim=-1)
            attn_out[:, :, i : i + 1, :] = torch.matmul(attn_prob, v_win)

        out = attn_out.transpose_(1,2).contiguous().view(b_q,l_q,self.dim)
        out = self.o_proj(out)

        return out
                


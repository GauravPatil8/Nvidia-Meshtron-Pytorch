import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.components.PositionalEncoding import RoPEncoding

class MultiHeadAttention(nn.Module):
    #vanilla attention block
    def __init__(self, dim: int, num_heads: int, dropout: float, rope_flag: bool = False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.dim_k = dim // num_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim)

        self.rope_flag = rope_flag


    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):

        batch_size, seq_len, dim = Q.shape
        _, n_k, _ = K.shape
        self.rope = RoPEncoding(dim, seq_len)

        Q = self.wq(Q)
        K = self.wk(K)
        V = self.wv(V)

        if self.rope_flag:
            Q = self.rope(Q)
            K = self.rope(K)

        Q = Q.view(batch_size, seq_len, self.heads, self.dim_k).transpose(1,2)
        K = K.view(batch_size, n_k, self.heads, self.dim_k).transpose(1,2)
        V = V.view(batch_size, n_k, self.heads, self.dim_k).transpose(1,2)

        attention_scores = (Q @ K.transpose(-2, -1)) * (math.sqrt(self.d_m))
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = self.dropout(F.softmax(attention_scores, dim = -1))

        out = attention_weights @ V

        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, dim)

        out = self.wo(out)

        return out
    
class SlidingWindowAttention(nn.Module):
    #Attends to tokens in a local window  
    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size
        self.dim_k = dim // num_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None):

        b, s, d = q.shape
        _, s_k, _ = k.shape
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        self.rope = RoPEncoding(d, s)
        q = self.rope(q)
        k = self.rope(k)
        
        q = q.view(b, s, self.heads, self.dim_k).transpose(1, 2)
        k = k.view(b, s_k, self.heads, self.dim_k).transpose(1, 2)
        v = v.view(b, s_k, self.heads, self.dim_k).transpose(1, 2)

        out = []
        for i in range(s):
            start = max(0, i - self.window_size)
            end = min(s, i + self.window_size + 1)

            q_i = q[:, :, i:i+1, :]
            k_i = k[:, :, start:end, :]
            v_i = v[:, :, start:end, :]

            attention_scores = (q_i @ (k_i.transpose(-2,-1))) * math.sqrt(self.dim_k)

            if mask:
                mask_window = mask[:, start:end].unsqueeze(1).unsqueeze(1)
                attention_scores = attention_scores.masked_fill(mask_window==0, float('-inf'))

            attention_scores = F.softmax(attention_scores, dim=-1)

            out_i = attention_scores @ v_i
            out.append(out_i)
        
        out = torch.cat(out, dim=2)
        out = out.transpose(1, 2).contiguous().view(b, s, d)

        return self.wo(out)
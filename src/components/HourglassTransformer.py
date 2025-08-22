import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class UpSample(nn.Module):
    def __init__(self, shorten_factor: int, dim: int, dropout:float):
        super().__init__()
        self.sf = shorten_factor
        self.dim = dim

        self.linear = nn.Linear(dim, shorten_factor * dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)

        b, s, _ = x.shape

        x = x.view(b, s * self.sf, self.dim)
        return x

class DownSample(nn.Module):
    def __init__(self, shorten_factor: int, dim: int, dropout:float):
        super().__init__()
        self.sf = shorten_factor
        self.dim = dim

        self.linear = nn.Linear(dim * shorten_factor , dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        b, s, d = x.shape
        assert d == self.dim, f"Expected dim={self.dim}, got {d}"
        assert s % self.sf == 0, f"Seq_len {s} not divisible by shorten_factor {self.sf}"

        x = x.view(b, s // self.sf, d * self.sf) 

        x = self.linear(x)
        x = self.dropout(x)

        return x
    
class InputEmbedding(nn.Module):
    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.num_tokens = num_tokens 
        self.dim = dim
        self.embedding = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dim)
    
class PositionalEncoding(nn.Module):
    # Rotary positional encoding (RoPE)
    def __init__(self, dim: int, seq_len: int):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len

        assert dim % 2 == 0, "Embedding dimension must be even"


        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        freqs = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000) / dim))

        angles = pos * freqs

        self.register_buffer('cos', torch.cos(angles), persistent=False)
        self.register_buffer('sin', torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.shape[1]

        if seq_len > self.seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum {self.seq_len}")
        
        even = x[..., ::2]
        odd = x[..., 1::2]

        cos = self.cos[:seq_len, :].unsqueeze(0).expand(x.shape[0], -1, -1)
        sin = self.sin[:seq_len, :].unsqueeze(0).expand(x.shape[0], -1, -1)

        x_pe = torch.cat([even * cos - odd * sin, even * sin + odd * cos], dim = -1)

        return x_pe
    
class MultiHeadSelfAttention(nn.Module):
    #vanilla attention block
    def __init__(self, dim: int, num_heads: int, dropout: float, seq_len: int):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.dim_k = dim // num_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim)

        self.rope = PositionalEncoding(dim, seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):

        batch_size, seq_len, _ = x.shape

        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        Q = self.rope(Q)
        K = self.rope(K)

        Q = Q.view(batch_size, seq_len, self.heads, self.dim_k).transpose(1,2)
        K = K.view(batch_size, seq_len, self.heads, self.dim_k).transpose(1,2)
        V = V.view(batch_size, seq_len, self.heads, self.dim_k).transpose(1,2)

        attention_scores = (Q @ K.transpose(-2, -1)) * (math.sqrt(self.d_m))
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = self.dropout(F.softmax(attention_scores, dim = -1))

        out = attention_weights @ V

        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, self.dim)

        out = self.wo(out)

        return out
    
class FeedForwardNetwork(nn.Module):
    pass

class LayerNormalization(nn.Module):
    def __init__(self, f_dim: int, eps: float=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(f_dim))
        self.beta = nn.Parameter(torch.zeros(f_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    
class ResidualConnection(nn.Module):
    #pre-norm residual connection
    def __init__(self, f_dim: int, dropout: float):
        super().__init__()
        self.f_dim = f_dim
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(f_dim)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class ProjectionLayer(nn.Module):
    def __init__(self, dim, num_tokens):
        super().__init__()
        self.proj = nn.Linear(dim, num_tokens)

    def forward(self, x):
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, 
                 f_dim: int, 
                 dropout: float,
                 attention_block: MultiHeadSelfAttention,
                 feed_forward_block: FeedForwardNetwork
                 ):
        super().__init__()
        self.norm = LayerNormalization(f_dim)
        self.residuals = nn.ModuleList(ResidualConnection(f_dim, dropout) for _ in range(2))
        self.dropout = dropout
        self.attention = attention_block
        self.FFN = feed_forward_block

    def forward(self, x, mask):
        x = self.residuals[0](x, lambda x: self.attention(x, mask))
        x = self.residuals[1](x, self.FFN)
        return x


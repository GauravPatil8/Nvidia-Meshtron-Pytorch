import torch
import torch.nn as nn
import math

class RoPEncoding(nn.Module):
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
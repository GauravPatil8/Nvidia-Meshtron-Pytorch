import torch
import torch.nn as nn
import math

class RoPEncoding(nn.Module):
    # Rotary positional encoding (RoPE)
    def __init__(self, dim: int, seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.theta = theta

        assert dim % 2 == 0, "Embedding dimension must be even"


        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)

        freqs = torch.exp(
            -torch.arange(0, dim, 2).float() * (math.log(theta) / dim)
            )

        angles = pos * freqs

        self.register_buffer('cos', torch.cos(angles), persistent=False)
        self.register_buffer('sin', torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.ndim == 3:
            _, seq_len, _ = x.shape
            cos = self.cos[:seq_len].unsqueeze(0)
            sin = self.sin[:seq_len].unsqueeze(0)

            even = x[..., ::2]
            odd = x[..., 1::2]

            return torch.cat([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
        

        if x.ndim == 4:
            _, _, seq_len, head_dim = x.shape
            assert head_dim % 2 == 0, "Head dimension must be even"

            cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,dim/2)
            sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)

            even, odd = x[..., ::2], x[..., 1::2]
            return torch.cat([even * cos - odd * sin,
                              even * sin + odd * cos], dim=-1)

        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        
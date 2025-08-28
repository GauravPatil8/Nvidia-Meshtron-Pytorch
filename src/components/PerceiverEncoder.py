import torch
import torch.nn as nn
from src.components.Attention import MultiHeadAttention

class ConditioningEncoder(nn.Module):
    """Conditions the meshtron model on 3d point cloud"""
    def __init__(self,
                 *,
                 num_latents:int,
                 latent_dim:int,
                 dim_ffn:int,
                 num_blocks:int,
                 heads:int,
                 num_self_attention: int):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.layers = nn.ModuleList([])
        self.self_attn = nn.ModuleList([])

        for _ in range(num_self_attention):
            self.self_attn.append([
                MultiHeadAttention(latent_dim, heads), # does self attention
                nn.Sequential(
                    nn.Linear(latent_dim, dim_ffn),
                    nn.GELU(),
                    nn.Linear(dim_ffn, latent_dim)
                ),
                nn.LayerNorm(latent_dim)
                ])
        for _ in range(num_blocks):
            block = [
                MultiHeadAttention(latent_dim, heads), # does cross attention
                nn.LayerNorm(latent_dim),
                nn.Sequential(
                    nn.Linear(latent_dim, dim_ffn),
                    nn.GELU(),
                    nn.Linear(dim_ffn, latent_dim)
                ),
                self.self_attn
            ]
            self.layers.append(block)
                
            
    def forward(self, inputs):
        b = inputs.shape[0]

        latents = self.latents.unsqueeze(0).expand(b,-1,-1)

        for cross_attn, norm1, cross_ffn,self_attn_blocks in self.layers:
            latents = latents + cross_attn(latents, inputs, inputs)
            latents = norm1(latents)
            latents = latents + cross_ffn(latents)
            
            for attn, ffn, norm in self_attn_blocks:
                latents = latents + attn(latents, latents, latents)
                latents = latents + ffn(latents)
                latents = norm(latents)

        return latents
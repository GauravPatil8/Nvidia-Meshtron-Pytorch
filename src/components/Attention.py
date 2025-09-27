# import math
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional, Tuple
# from src.components.RollingKV import RollingKVCache

# class FlashAttention(nn.Module):
#     """
#     FlashAttention2 implementation with tiling and recomputation for memory efficiency.
    
#     This implementation demonstrates the key concepts of FlashAttention2:
#     - Block-wise computation to reduce memory footprint
#     - Online softmax computation
#     - Causal masking support
#     - Dropout support
#     """
    
#     def __init__(
#         self,
#         head_dim: int,
#         dropout: float = 0.0,
#         scale: Optional[float] = None,
#         block_size: int = 64
#     ):
#         super().__init__()
#         self.head_dim = head_dim
#         self.dropout = dropout
#         self.scale = scale if scale is not None else head_dim ** -0.5
#         self.block_size = block_size
#         self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
#     def forward(
#         self,
#         q: torch.Tensor,
#         k: torch.Tensor,
#         v: torch.Tensor,
#         causal: bool = False,
#         return_softmax: bool = False,
#         kv_cache: Optional[RollingKVCache] = None,
#         layer_idx: Optional[int] = None
#         ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         """
#         Forward pass of FlashAttention2.
        
#         Args:
#             q: Query tensor [batch_size, num_heads, seq_len, head_dim]
#             k: Key tensor [batch_size, num_heads, seq_len, head_dim]
#             v: Value tensor [batch_size, num_heads, seq_len, head_dim]
#             causal: Whether to apply causal masking
#             return_softmax: Whether to return attention weights (memory intensive)
            
#         Returns:
#             output: Attention output [batch_size, num_heads, seq_len, head_dim]
#             attention_weights: Optional attention weights if return_softmax=True
#         """

#         batch_size, num_heads, seq_len, head_dim = q.shape


#         if kv_cache is not None and layer_idx is not None:

#             # Convert current KV from [batch_size, num_heads, seq_len, head_dim] 
#             # to [seq_len, num_heads, head_dim] for cache update
#             k_new = k.unsqueeze(0).transpose(0, 1)
#             v_new = v.unsqueeze(0).transpose(0, 1)

#             kv_cache.update(layer_idx, k_new, v_new)

#             k_full, v_full = kv_cache.get(layer_idx)

#             # Convert back to [batch_size, num_heads, full_len, head_dim]
#             k = k_full.transpose(0, 1).unsqueeze(0)
#             v = v_full.transpose(0, 1).unsqueeze(0)

#             full_seq_len = k.size(2)
#         else:
#             full_seq_len = seq_len

        
#         # Scale queries
#         q = q * self.scale
        
#         # Initialize output tensor
#         o = torch.zeros_like(q)
        
#         # Initialize LSE (log-sum-exp) for stable softmax computation
#         lse = torch.zeros(batch_size, num_heads, seq_len, device=q.device, dtype=torch.float32)
        
#         # Block-wise computation
#         num_q_blocks = math.ceil(seq_len / self.block_size)
#         num_kv_blocks = math.ceil(full_seq_len / self.block_size)

        
#         # Optional: store attention weights if requested
#         # if return_softmax:
#         #     attn_weights = torch.zeros(
#         #         batch_size, num_heads, seq_len, seq_len, 
#         #         device=q.device, dtype=q.dtype
#         #     )
        
#         for i in range(num_q_blocks):
#             # Query block indices
#             q_start = i * self.block_size
#             q_end = min((i + 1) * self.block_size, seq_len)
            
#             # Get query block
#             q_block = q[:, :, q_start:q_end]
            
#             # Initialize block outputs
#             o_block = torch.zeros_like(q_block)
#             lse_block = torch.full(
#                 (batch_size, num_heads, q_end - q_start),
#                 float('-inf'),
#                 device=q.device,
#                 dtype=torch.float32
#             )
            
#             # Iterate over key-value blocks
#             for j in range(num_kv_blocks):
#                 # Key-value block indices
#                 kv_start = j * self.block_size
#                 kv_end = min((j + 1) * self.block_size, seq_len)
                
#                 # Skip blocks that are masked out in causal attention
#                 # if causal > q_end - 1:
#                 #     break
                
#                 # Get key-value blocks
#                 k_block = k[:, :, kv_start:kv_end]
#                 v_block = v[:, :, kv_start:kv_end]
                
#                 # Compute attention scores for this block
#                 scores = torch.matmul(q_block, k_block.transpose(-2, -1))
                
#                 # Apply causal mask if needed
#                 if causal:
#                     block_mask = self._create_causal_mask(
#                         q_end - q_start, 
#                         kv_end - kv_start,
#                         q_start, 
#                         kv_start,
#                         device=scores.device
#                     )
#                     scores = scores.masked_fill(block_mask, float('-inf'))
                
#                 # Online softmax computation
#                 block_lse_new = torch.logsumexp(scores, dim=-1)
                
#                 # Combine with previous LSE values
#                 lse_old = lse_block
#                 lse_new = torch.logaddexp(lse_old, block_lse_new)
                
#                 # Compute attention weights for this block
#                 attn_block = torch.exp(scores - lse_new.unsqueeze(-1))
                
#                 # Apply dropout if specified
#                 if self.dropout_layer is not None and self.training:
#                     attn_block = self.dropout_layer(attn_block)
                
#                 # Store attention weights if requested
#                 if return_softmax:
#                     attn_weights[:, :, q_start:q_end, kv_start:kv_end] = attn_block
                
#                 # Update output with rescaling
#                 rescale = torch.exp(lse_old - lse_new)
#                 o_block = o_block * rescale.unsqueeze(-1) + torch.matmul(attn_block, v_block)
                
#                 # Update LSE
#                 lse_block = lse_new
            
#             # Write block results back
#             o[:, :, q_start:q_end] = o_block
#             lse[:, :, q_start:q_end] = lse_block
        
#         if return_softmax:
#             return o, attn_weights
#         return o, None
    
#     def _create_causal_mask(
#         self, 
#         q_len: int, 
#         kv_len: int, 
#         q_offset: int, 
#         kv_offset: int,
#         device: torch.device
#     ) -> torch.Tensor:
#         """Create a causal mask for a block of attention scores."""
#         mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=device)
#         for i in range(q_len):
#             for j in range(kv_len):
#                 if q_offset + i < kv_offset + j:
#                     mask[i, j] = True
#                 else:
#                     mask[i, j] = False
#         return mask


# class MultiHeadFlashAttention(nn.Module):
#     """
#     Multi-head attention using FlashAttention2.
#     """
    
#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         dropout: float = 0.0,
#         bias: bool = True,
#         block_size: int = 64
#     ):
#         super().__init__()
#         assert embed_dim % num_heads == 0
        
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
        
#         # Linear projections
#         self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
#         # FlashAttention2
#         self.flash_attn = FlashAttention(
#             head_dim=self.head_dim,
#             dropout=dropout,
#             block_size=block_size
#         )
        
#     def forward(
#         self,
#         q: torch.Tensor,
#         k: torch.Tensor,
#         v: torch.Tensor,
#         causal: bool = True,
#         return_attention: bool = False,
#         kv_cache: Optional[RollingKVCache] = None,
#         layer_idx: Optional[int] = None
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         """
#         Forward pass of multi-head FlashAttention2.
        
#         Args:
#             x: Input tensor [batch_size, seq_len, embed_dim]
#             causal: Whether to apply causal masking
#             return_attention: Whether to return attention weights
            
#         Returns:
#             output: Output tensor [batch_size, seq_len, embed_dim]
#             attention_weights: Optional attention weights
#         """
#         q_batch_size, q_seq_len, q_embed_dim = q.shape
#         k_batch_size, k_seq_len, k_embed_dim = k.shape
#         v_batch_size, v_seq_len, v_embed_dim = v.shape

        
#         # Linear projections and reshape
#         q = self.q_proj(q).view(q_batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         k = self.k_proj(k).view(k_batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         v = self.v_proj(v).view(v_batch_size, v_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
#         # Apply FlashAttention2
#         attn_output, attn_weights = self.flash_attn(q, k, v, causal=causal, return_softmax=return_attention,kv_cache = kv_cache, layer_idx = layer_idx)
        
#         # Reshape and apply output projection
#         attn_output = attn_output.transpose(1, 2).contiguous().view(q_batch_size, q_seq_len, q_embed_dim)
#         output = self.out_proj(attn_output)
        
#         return output, attn_weights

import torch
import torch.nn as nn
from flash_attn import flash_attn_func


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0,bias = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model * 3, bias=bias)
        self.kv_proj = nn.Linear(d_model, d_model * 3, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, q, kv):
        q_batch_size, q_seq_len, _ = q.shape
        kv_batch_size, kv_seq_len, kv_dim = kv.shape

        q = self.q_proj(q)
        kv = self.kv_proj(kv)

        kv = kv.reshape(kv_batch_size, kv_seq_len, 2, self.num_heads, self.num_heads // kv_dim)

        k, v = kv.unbind(dim = 2)

        output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=None,  # Uses 1/sqrt(head_dim) by default
            causal=True,  # Set to True for causal masking (like in GPT)
        )

        output = output.reshape(q_batch_size, q_seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(output)
        
        return output
import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from typing import Optional
from src.components.RollingKV import RollingKVCache
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=bias, dtype=torch.float16)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, dtype=torch.float16)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, dtype=torch.float16)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias, dtype=torch.float16)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):
        q = q.to(dtype=torch.float16)
        k = k.to(dtype=torch.float16)
        v = v.to(dtype=torch.float16)
        q_batch_size, q_seq_len, _ = q.shape
        k_batch_size, k_seq_len, _ = k.shape
        v_batch_size, v_seq_len, _ = v.shape


        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(q_batch_size, q_seq_len, self.num_heads, self.head_dim)
        k = k.view(k_batch_size, k_seq_len, self.num_heads, self.head_dim)
        v = v.view(v_batch_size, v_seq_len, self.num_heads, self.head_dim)

        output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=None,
            causal=True
        )

        output = output.view(q_batch_size, q_seq_len, self.d_model)
        output = self.out_proj(output)
        
        return output
    

class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention mechanism where each token attends only to 
    tokens within a fixed window size around it.
    
    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        window_size: Size of the sliding window (tokens on each side)
        dropout: Dropout probability
    """
    def __init__(self, d_model, n_heads, window_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.scale = torch.sqrt(self.d_head)
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def create_sliding_window_mask(self, seq_len, device):
        """Create a mask for sliding window attention"""
        # Create position indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        # Calculate distance between all pairs
        distances = torch.abs(positions.T - positions)
        # Mask out positions outside the window
        mask = distances <= self.window_size
        return mask
    
    def forward(self,q, k, v, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional additional mask (e.g., padding mask)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        B, L, D = q.shape
        k_b, k_l, K_d = k.shape
        v_b, v_l, v_d = v.shape

        # Linear projections and reshape for multi-head attention
        q = self.q_proj(q).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(k).view(k_b, k_l, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(v).view(v_b, v_l, self.n_heads, self.d_head).transpose(1, 2)
        # Shape: (B, n_heads, L, d_head)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # Shape: (B, n_heads, L, L)
        
        # Create sliding window mask
        window_mask = self.create_sliding_window_mask(L, q.device)
        # Shape: (L, L)
        
        # Apply sliding window mask
        attn_scores = attn_scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        # Shape: (B, n_heads, L, d_head)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        # Shape: (B, L, d_model)
        
        # Final linear projection
        out = self.out_proj(out)
        
        return out
        
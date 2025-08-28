import torch 
import torch.nn as nn
from src.components.HourglassTransformer import (
    Transformer,
    InputEmbedding,
    ProjectionLayer,
    FeedForwardNetwork,
    parse_hierarchy,
    build_hourglass_valley,
    SwiGLU
)
from src.components.Attention import MultiHeadAttention

class Meshtron(nn.Module):
    def __init__(self,
                 dim: int,
                 vocab_size: int,
                 n_heads: int,
                 attn_window_size: int,
                 d_ff: int,
                 hierarchy: str,
                 dropout: float
                 ):
        super().__init__()
        shortening_factor, num_blocks, n_pre_post_blocks = parse_hierarchy(hierarchy=hierarchy)
        self.sf = shortening_factor
        self.n_blocks = num_blocks
        self.embedding = InputEmbedding(vocab_size, dim)
        self.pre_blocks = nn.ModuleList([
            Transformer(dim, 
                        dropout, 
                        MultiHeadAttention(dim, n_heads, dropout, rope_flag=True), 
                        FeedForwardNetwork(dim, d_ff, dropout, SwiGLU)
            ) for _ in range(n_pre_post_blocks)
        ])
        self.valley = build_hourglass_valley(
            dim,
            n_heads,
            attn_window_size,
            vocab_size,
            self.sf,
            self.n_blocks,
            d_ff,
            dropout
        )
                                             
        self.post_block = nn.ModuleList([
            Transformer(dim, 
                        dropout, 
                        MultiHeadAttention(dim, n_heads, dropout, rope_flag=True), 
                        FeedForwardNetwork(dim, d_ff, dropout, SwiGLU)
            ) for _ in range(n_pre_post_blocks)
        ])
        self.out_proj = ProjectionLayer(dim, vocab_size)

    def forward(self, x, mask):
        x = self.embedding(x)

        for block in self.pre_blocks:
            x = block(x, mask)

        for layer in self.valley:
            x = layer(x, mask)

        for block in self.post_blocks:
            x = block(x, mask)
        
        return self.out_proj(x)
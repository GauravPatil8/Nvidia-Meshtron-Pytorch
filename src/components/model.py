import torch.nn as nn
import torch.nn.functional as F 
from src.components.HourglassTransformer import (
    Transformer,
    InputEmbedding,
    ProjectionLayer,
    FeedForwardNetwork,
    LinearUpSample,
    parse_hierarchy,
    build_hourglass_valley,
    SwiGLU
)
from src.components.Attention import MultiHeadAttention
from src.components.PerceiverEncoder import ConditioningEncoder


class Meshtron(nn.Module):
    def __init__(self,
                 *,
                 dim: int,
                 vocab_size: int,
                 n_heads: int,
                 attn_window_size: int,
                 d_ff: int,
                 hierarchy: str,
                 dropout: float,
                 use_conditioning: bool,
                 con_num_latents:int,
                 con_latent_dim:int,
                 con_dim_ffn:int,
                 con_num_blocks:int,
                 con_n_attn_heads:int,
                 con_num_self_attention_blocks: int,
                 condition_every_n_layers: int,
                 ):
        super().__init__()
        shortening_factor, num_blocks, n_pre_post_blocks = parse_hierarchy(hierarchy=hierarchy)
        self.sf = shortening_factor
        self.n_blocks = num_blocks
        self.embedding = InputEmbedding(vocab_size, dim)
        self.up_sample = LinearUpSample(shortening_factor)
        self.use_conditioning = use_conditioning
        self.point_cloud_conditioning = ConditioningEncoder(
            num_latents=con_num_latents,
            latent_dim=con_latent_dim,
            dim_ffn=con_dim_ffn,
            num_blocks=con_num_blocks,
            heads=con_n_attn_heads,
            num_self_attention=con_num_self_attention_blocks
        )

        self.pre_blocks = nn.ModuleList([
            Transformer(
                dim,
                dropout,
                MultiHeadAttention(dim, n_heads, dropout, rope_flag=True),
                FeedForwardNetwork(dim, d_ff, dropout, SwiGLU),
                conditioning_flag= use_conditioning and (i % condition_every_n_layers == 0)
            )
            for i in range(n_pre_post_blocks)
        ])

        self.down_valley, self.center_layer, self.up_valley = build_hourglass_valley(
            dim,
            n_heads,
            attn_window_size,
            vocab_size,
            self.sf,
            self.n_blocks,
            d_ff,
            dropout,
            condition_every_n_layers,
            use_conditioning
        )
                                             
        self.post_block = nn.ModuleList([
            Transformer(dim, 
                        dropout, 
                        MultiHeadAttention(dim, n_heads, dropout, rope_flag=True), 
                        FeedForwardNetwork(dim, d_ff, dropout, SwiGLU),
                        conditioning_flag= use_conditioning and (i % condition_every_n_layers == 0)
            ) for i in range(n_pre_post_blocks)
        ])
        self.out_proj = ProjectionLayer(dim, vocab_size)

    def forward(self, x, conditioning_data, mask):

        skips = [] #holds skip connection values, used in upsampling
        x = self.embedding(x)

        #conditioning tensor
        cond= None

        # Pre valley block
        for block in self.pre_blocks:
            cond = self.point_cloud_conditioning(conditioning_data) if block.use_conditioning else None
            x = block(x=x, conditions=cond, mask=mask)
        skips.append(x) # Appending residuals to be added later

        #Downsampling valley
        for block in self.down_valley:
            cond = self.point_cloud_conditioning(conditioning_data) if block.use_conditioning else None
            x = block(x=x, conditions=cond, mask=mask)
            skips.append(x)

        #center layer of valley
        x = self.center_layer(x = x, conditions = cond, mask = mask)

        
        #upsampling valley
        for block, skip in zip(self.up_valley, reversed(skips)):
            x = self.up_sample(x, skip)
            cond = self.point_cloud_conditioning(conditioning_data) if block.use_conditioning else None
            x = block(x=x, conditions=cond, mask=mask)

        #upsampling for the last vanilla block
        x = self.up_sample(x, skips[0])

        #last vanilla blocks
        for block in self.post_blocks:
            cond = self.point_cloud_conditioning(conditioning_data) if block.use_conditioning else None
            x = block(x=x, conditions=cond, mask=mask)
        
        return self.out_proj(x)
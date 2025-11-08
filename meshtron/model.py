import torch
import torch.nn as nn
from meshtron.encoder_conditioning import ConditioningEncoder
from meshtron.decoder_hourglass import (
    Transformer,
    InputEmbedding,
    ProjectionLayer,
    LinearUpSample,
    parse_hierarchy,
    build_hourglass_valley
)

class Meshtron(nn.Module):
    def __init__(self,
                 *,
                 dim: int,
                 embedding_size: int,
                 n_heads: int,
                 head_dim: int,
                 window_size: int,
                 d_ff: int,
                 hierarchy: str,
                 dropout: float,
                 pad_token: int,
                 condition_every_n_layers:int,
                 encoder: ConditioningEncoder,
                 ):
        super().__init__()

        shortening_factor_list, num_blocks, n_pre_post_blocks = parse_hierarchy(hierarchy=hierarchy)


        self.sf = shortening_factor_list
        self.n_blocks = num_blocks
        
        self.embedding = InputEmbedding(embedding_size, dim)
        self.up_sample = LinearUpSample(shortening_factor_list[0], dim)
        self.conditioning_encoder = encoder

        self.pre_blocks = nn.ModuleList([
            Transformer(dim,n_heads,head_dim,d_ff,window_size,dropout,conditioning_flag= (((i+1) % condition_every_n_layers) == 0) and i != 0) 
            for i in range(n_pre_post_blocks)
        ])

        self.down_valley, self.center_layer, self.up_valley = build_hourglass_valley(
            dim=dim,
            num_of_heads=n_heads,
            head_dim=head_dim,
            h_sfs=self.sf,
            h_nl=self.n_blocks,
            d_ff=d_ff,
            window_size=window_size,
            dropout=dropout,
            pad_token=pad_token,
            condition_every_n_layers=condition_every_n_layers,
        )
                                             
        self.post_block = nn.ModuleList([
            Transformer(dim,n_heads,head_dim,d_ff,window_size,dropout,conditioning_flag= (((i+1) % condition_every_n_layers) == 0) and i != 0) 
            for i in range(n_pre_post_blocks)
        ])

        self.out_proj = ProjectionLayer(dim, embedding_size)
        
    
    def forward(self, data, conditioning_data, face_count, quad_ratio, mask):

        #conditioning tensor
        cond = self.conditioning_encoder(conditioning_data, face_count, quad_ratio)

        def run_block(block: Transformer, data):
            conditions = cond if block.conditioning_flag else None
            return block(x = data, conditions=conditions, mask=mask)
        
        skips = [] #holds skip connection values, used in upsampling
        data = self.embedding(data)

        # Pre valley block
        for block in self.pre_blocks:
            data = run_block(block, data)
        skips.append(data) # Appending residuals to be added later

        #Downsampling valley
        for layer in self.down_valley:
            data = layer(x=data, conditions=cond, mask=mask)
            skips.append(data)

        #center layer of valley
        data = self.center_layer(x = data, conditions = cond, mask = mask)

        #upsampling valley
        for layer, skip in zip(self.up_valley, reversed(skips[1:])):
            data = self.up_sample(data) + skip
            data = layer(x=data, conditions=cond, mask=mask)

        #upsampling for the last vanilla block
        data = self.up_sample(data) + skips[0]

        #last vanilla blocks
        for block in self.post_block:
            data = run_block(block, data)
        
        return data
        
    def project(self, x: torch.Tensor):
        return self.out_proj(x)

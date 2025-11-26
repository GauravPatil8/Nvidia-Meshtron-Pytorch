import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from meshtron.encoder_conditioning import ConditioningEncoder
from meshtron.decoder_hourglass import (
    InputEmbedding,
    ProjectionLayer,
    LinearUpSample,
    LinearDownSample,
    build_hourglass_valley,
    pad_to_multiple
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
                 shortening_factor: int,
                 num_blocks_per_layers: List[int],
                 ff_dropout: float,
                 attn_dropout:float,
                 pad_token: int,
                 condition_every_n_layers:int,
                 encoder: ConditioningEncoder,
                 ):
        super().__init__()
        self.sf = shortening_factor
        self.embedding = InputEmbedding(embedding_size, dim)
        self.up_sample = LinearUpSample(shortening_factor, dim)
        self.down_sample = LinearDownSample(shortening_factor, dim)
        self.total_reduction = len(num_blocks_per_layers) - 1
        self.pad_token = pad_token
        self.conditioning_encoder = encoder
        self.out_proj = ProjectionLayer(dim, embedding_size)
        
        self.pre_layer, self.down_valley, self.center_layer, self.up_valley, self.post_layer = build_hourglass_valley(
            dim=dim,
            num_of_heads=n_heads,
            head_dim=head_dim,
            sf=shortening_factor,
            num_blocks=num_blocks_per_layers,
            d_ff=d_ff,
            window_size=window_size,
            ff_dropout=ff_dropout,
            attn_dropout = attn_dropout,
            rope=RotaryEmbedding(dim=head_dim),
            condition_every_n_layers=condition_every_n_layers,
        )

    def __causal_upsample(self, x):
        x = self.up_sample(x)
        shift = self.sf - 1
        x = F.pad(x, (0, 0, shift, -shift), value=0.) #padding for preventing leak
        return x
    
    def forward(self, data, conditioning_data, face_count, quad_ratio, mask):

        skips = [] #holds skip connection values, used in upsampling

        #conditioning tensor
        cond = self.conditioning_encoder(conditioning_data, face_count, quad_ratio)

        data = pad_to_multiple(data, self.sf ** self.total_reduction, dim=-1, value=self.pad_token)
        pad_mask = (data == self.pad_token)
        data = self.embedding(data)
        data = data.masked_fill(pad_mask.unsqueeze(-1), 0.0)#zeroing pad tokens

        # Pre valley block
        data = self.pre_layer(x=data, conditions = cond, mask = mask)
        skips.append(data) # Appending residuals to be added later

        ##Down valley
        data = self.down_sample(data)
        data = self.down_valley(x=data, conditions=cond, mask=mask)
        skips.append(data)

        #center layer of valley
        data = self.down_sample(data)
        data = self.center_layer(x = data, conditions = cond, mask = mask)

        #upsampling valley
        data = self.__causal_upsample(data) + skips[-1]
        data = self.up_valley(x=data, conditions=cond, mask=mask)

        #upsampling for the last vanilla block(post layer)
        data = self.__causal_upsample(data) + skips[0]
        data = self.post_layer(x=data, conditions = cond, mask = mask)
        
        return data
        
    def project(self, x: torch.Tensor):
        return self.out_proj(x)

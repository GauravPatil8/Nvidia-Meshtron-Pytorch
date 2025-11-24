import torch
import math
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from meshtron._attention import Attention
from rotary_embedding_torch import RotaryEmbedding

def pad_to_multiple(tensor, multiple, dim = -1, value = 0):
        seq_len = tensor.shape[dim]
        m = seq_len / multiple
        if m.is_integer():
            return tensor
        remainder = math.ceil(m) * multiple - seq_len
        pad_offset = (0,) * (-1 - dim) * 2
        return F.pad(tensor, (*pad_offset, 0, remainder), value = value)


def SwiGLU(x: torch.Tensor):
    "SwiGLU activation function"
    x1, x2 = x.chunk(2, dim=-1)
    return F.silu(x2) * x1
        
class LinearUpSample(nn.Module):
    def __init__(self, shorten_factor: int, dim: int):
        super().__init__()
        self.sf = shorten_factor
        self.dim = dim
        self.linear = nn.Linear(dim, shorten_factor * dim)

    def forward(self, x):
        b, s, _ = x.shape
        x = self.linear(x)
        x = x.view(b, s * self.sf, self.dim)
        return x

class LinearDownSample(nn.Module):
    def __init__(self, shorten_factor: int, dim: int):
        super().__init__()
        self.sf = shorten_factor
        self.dim = dim
        self.linear = nn.Linear(dim*shorten_factor, dim)

    def forward(self, x):
        b, s, _ = x.shape
        return self.linear(x.view(b, s // self.sf, self.dim*self.sf))
    
class InputEmbedding(nn.Module):
    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.num_tokens = num_tokens 
        self.dim = dim
        self.embedding = nn.Embedding(num_tokens, dim)
        self.scale = math.sqrt(dim)

    def forward(self, x):
        return self.embedding(x).mul_(self.scale)
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, dim:int, d_ff:int, dropout:float, activation):
        super().__init__()
        self.linear1 = nn.Linear(dim, 2 * d_ff) #for swiglu chunking
        self.linear2 = nn.Linear(d_ff, dim)
        self.dropout = nn.Dropout(dropout, inplace = True)
        self.activation = activation
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
    
class ResidualConnection(nn.Module):
    #pre-norm residual connection
    def __init__(self, f_dim: int, dropout: float):
        super().__init__()
        self.f_dim = f_dim
        self.dropout = nn.Dropout(dropout, inplace = True)
        self.norm = nn.LayerNorm(f_dim, bias = False)

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
                 dim: int,
                 num_heads:int,
                 head_dim: int,
                 dim_ff: int,
                 window_size:int, 
                 ff_dropout: float,
                 attn_dropout:float,
                 rope: RotaryEmbedding,
                 conditioning_flag: bool = False,
                 ):
        super().__init__()
        self.conditioning_flag = conditioning_flag
        if conditioning_flag:
            self.residuals = nn.ModuleList([ResidualConnection(dim, ff_dropout) for _ in range(3)])
        else:
            self.residuals = nn.ModuleList([ResidualConnection(dim, ff_dropout) for _ in range(2)])

        self.dropout = ff_dropout
        self.attention = Attention(dim, num_heads, head_dim, window_size, rope, attn_dropout)
        self.FFN = FeedForwardNetwork(dim, dim_ff, ff_dropout, SwiGLU)

    def forward(self,*, x: torch.Tensor, conditions: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None):
        x = self.residuals[0](x, lambda x: self.attention(q=x,k=x, v=x, mask=mask))
        if self.conditioning_flag:
            x = self.residuals[1](x, lambda x: self.attention(q=x,k= conditions, v=conditions, mask=mask))
            x = self.residuals[2](x, self.FFN)
        else:
            x = self.residuals[1](x, self.FFN)

        return x

class Layer(nn.Module):
    def __init__(self,
                 *,
                 dim: int,
                 ff_dropout: float,
                 attn_dropout: float,
                 n_heads: int,
                 head_dim: int,
                 num_blocks:int,
                 d_ff: int,
                 window_size:int,
                 rope: RotaryEmbedding,
                 condition_every_n_layers: bool = False
                 ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Transformer(
                dim,
                n_heads,
                head_dim,
                d_ff,
                window_size,
                ff_dropout,
                attn_dropout,
                rope,
                conditioning_flag=(((i+1) % condition_every_n_layers) == 0)
            ) for i in range(num_blocks)
        ])

    def forward(self, x, conditions, mask):
        #tranformer blocks   
        for block in self.blocks:
            x = block(x = x, conditions = conditions, mask = mask)
        
        return x
        

def build_hourglass_valley(
        dim:int,
        num_of_heads: int,
        head_dim: int,
        sf: int,
        num_blocks: list[int],
        d_ff: int,
        window_size:int,
        ff_dropout:float,
        attn_dropout:float,
        rope: RotaryEmbedding,
        condition_every_n_layers: bool
    ) -> nn.ModuleList:
    
    assert len(num_blocks) == 3

    layer_config = dict(
        dim=dim,
        ff_dropout=ff_dropout,
        attn_dropout=attn_dropout,
        n_heads=num_of_heads,
        head_dim=head_dim,
        d_ff=d_ff,
        window_size=window_size,
        rope = rope,
        condition_every_n_layers=condition_every_n_layers,
    )

    pre_post_blocks_num = num_blocks[0]
    down_blocks_num = num_blocks[1]
    centre_blocks_num = num_blocks[2]


    pre_layer = Layer(num_blocks=pre_post_blocks_num, **layer_config)
    down_valley = Layer(num_blocks=down_blocks_num, **layer_config)
    center_layer = Layer(num_blocks=centre_blocks_num, **layer_config)
    up_valley = Layer(num_blocks=down_blocks_num, **layer_config)
    post_layer = Layer(num_blocks=pre_post_blocks_num, **layer_config)

    return pre_layer, down_valley, center_layer, up_valley, post_layer

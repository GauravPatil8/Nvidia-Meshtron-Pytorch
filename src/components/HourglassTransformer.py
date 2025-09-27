import torch
import math
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from src.components.RollingKV import RollingKVCache
from src.components.Attention import MultiHeadFlashAttention
def parse_hierarchy(hierarchy: str):
    """Parse hierarchy of the entire hourglass transformer.

    Parameters:
        hierarchy (str): A space-separated string specifying the transformer hierarchy.
                        Format: "pre-block funnel post-block, layer: n_blocks @ shortening factor (2@1)"
                        Example: "2@1 4@2 6@4 8@8 6@4 4@2 2@1"
                            - pre-block  : "2@1"
                            - funnel     : "4@2 6@4 8@8 6@4 4@2"
                            - post-block : "2@1"

    Returns:
        - relative_shortening_factor (list[int]) - "List of relative scale factors for only the funnel levels."
        - n_blocks (list[int]) - "List of number of blocks at each funnel level."
        - number_of_pre_post_blocks (int) - " Number of blocks in the pre and post blocks."
    """

    levels = hierarchy.split(' ')
    if levels != levels[::-1]:
        raise ValueError("Hierarchy ain't right brev")
    layers = [(x.split('@')) for x in levels[:1 + (len(levels) // 2)]]
    n_blocks = [int(x[0]) for x in layers]
    total_sf = [int(x[1]) for x in layers]

    relative_sf = []
    for current, prev in zip(total_sf, [1] + total_sf[:-1]):
        if current % prev !=0:
            raise ValueError(f"Aye brev fix your hierarchy,cause Hierarchy is not divisible by previous level: {current}, {prev}")
        relative_sf.append(current // prev)

    return relative_sf[1:], n_blocks[1:], n_blocks[0]

def SwiGLU(x: torch.Tensor):
    "SwiGLU activation function"

    x1, x2 = x.split(2, dim=-1)
    return F.silu(x1) * x2

class LinearUpSample(nn.Module):
    def __init__(self, shorten_factor: int, dim: int):
        super().__init__()
        self.sf = shorten_factor
        self.dim = dim

        self.linear = nn.Linear(dim, shorten_factor * dim)

    def forward(self, x, skip):
        shift = self.sf - 1
        x = F.pad(x, (0,0,shift, -shift), value=0.) #causal padding for preventing leak
        x = self.linear(x)
        b, s, _ = x.shape

        x = x.view(b, s * self.sf, self.dim)
        x = x + skip
        return x

class LinearDownSample(nn.Module):
    def __init__(self, shorten_factor: int, dim: int):
        super().__init__()
        self.sf = shorten_factor
        self.dim = dim
        self.linear = nn.Linear(dim*shorten_factor, dim)
        
    def forward(self, x):
        x = self.linear(x)
        b, s, _ = x.shape
        x = x.view(b, s // self.sf, self.dim*self.sf)
    
class InputEmbedding(nn.Module):
    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.num_tokens = num_tokens 
        self.dim = dim
        self.embedding = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dim)
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, dim:int, d_ff:int, dropout:float, activation):
        super().__init__()
        self.linear1 = nn.Linear(dim, 2 * d_ff) #for swiglu chunking
        self.linear2 = nn.Linear(2 * d_ff, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear2(x)


class LayerNormalization(nn.Module):
    def __init__(self, f_dim: int, eps: float=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(f_dim))
        self.beta = nn.Parameter(torch.zeros(f_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    
class ResidualConnection(nn.Module):
    #pre-norm residual connection
    def __init__(self, f_dim: int, dropout: float):
        super().__init__()
        self.f_dim = f_dim
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(f_dim)

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
                 f_dim: int, 
                 dropout: float,
                 attention_block: MultiHeadFlashAttention,
                 feed_forward_block: FeedForwardNetwork,
                 conditioning_flag: bool = False,
                 ):
        super().__init__()
        self.norm = LayerNormalization(f_dim)
        self.conditioning_flag = conditioning_flag
        if conditioning_flag:
            self.residuals = nn.ModuleList([ResidualConnection(f_dim, dropout) for _ in range(3)])
        else:
            self.residuals = nn.ModuleList([ResidualConnection(f_dim, dropout) for _ in range(2)])

        self.dropout = dropout
        self.attention = attention_block
        self.FFN = feed_forward_block

    def forward(self,*, x: torch.Tensor, conditions: torch.Tensor, mask: torch.Tensor, rolling_kv_cache: Optional[RollingKVCache] = None ):
        x = self.residuals[0](x, lambda x: self.attention(q=x,k= x,v= x, kv_cache=rolling_kv_cache))
        if self.conditioning_flag:
            x = self.residuals[1](x, lambda x: self.attention(q=x,k= conditions,v= conditions, kv_cache=rolling_kv_cache))
            x = self.residuals[2](x, self.FFN)
        else:
            x = self.residuals[1](x, self.FFN)

        return x

class Layer(nn.Module):

    def __init__(self,
                 *,
                 dim: int,
                 shortening_factor: int,
                 dropout: float,
                 n_heads: int,
                 block_size: int,
                 num_blocks:int,
                 d_ff: int,
                 downflag: bool = False,
                 condition_every_n_layers: bool = False,
                 use_conditioning:bool = False,
                 rolling_kv_cache: Optional[RollingKVCache] = None):
        super().__init__()
        self.sf = shortening_factor
        self.dropout = dropout
        self.norm = LayerNormalization(dim)
        self.downsample = LinearDownSample(dim, self.sf)
        self.residuals = ResidualConnection(dim, dropout)
        self.downflag = downflag
        self.blocks = nn.ModuleList([
            Transformer(dim,
                        dropout,
                        MultiHeadFlashAttention(dim, n_heads, dropout, False, block_size),
                        FeedForwardNetwork(dim, d_ff, dropout, SwiGLU),
                        conditioning_flag = use_conditioning and (i % condition_every_n_layers == 0) and i != 0,
                        )
            for i in range(num_blocks)
        ])
        self.rolling_kv_cache = rolling_kv_cache

    def forward(self, x, conditions, mask):
        
        if self.downflag:
            x = self.downsample(x)

        #tranformer blocks   
        def run_blocks(x, conditions, mask, rolling_kv_cache):
            for block in self.blocks:
                x = block(x = x, conditions = conditions, mask = mask, rolling_kv_cache = rolling_kv_cache)
            return x

        x = self.residuals(x, lambda x: run_blocks(x, conditions, mask, self.rolling_kv_cache))

        return x
        

def build_hourglass_valley(
        dim:int,
        num_of_heads: int,
        block_size: int,
        h_sfs: list[int],
        h_nl: list[int],
        d_ff: int,
        dropout: float,
        condition_every_n_layers: bool,
        use_conditioning: bool,
        rolling_kv_cache: Optional[RollingKVCache] = None
    ) -> nn.ModuleList:
    
    assert len(h_sfs) == len(h_nl) >= 1

    down_layers = nn.ModuleList()

   #funneling down
    for sf, n_layers in list(zip(h_sfs, h_nl))[:-1]:
        layer = Layer(
            dim=dim,
            shortening_factor=sf,
            dropout=dropout,
            n_heads=num_of_heads,
            block_size=block_size,
            num_blocks=n_layers,
            d_ff=d_ff,
            downflag=True,
            condition_every_n_layers=condition_every_n_layers,
            use_conditioning=use_conditioning,
            rolling_kv_cache=rolling_kv_cache
        )
        down_layers.append(layer)

    #middle layer
    center_layer = Layer(
            dim=dim,
            shortening_factor=h_sfs[-1],
            dropout=dropout,
            n_heads=num_of_heads,
            block_size=block_size,
            num_blocks=h_nl[-1],
            d_ff=d_ff,
            downflag=True,
            condition_every_n_layers=condition_every_n_layers,
            use_conditioning=use_conditioning,
            rolling_kv_cache=rolling_kv_cache
        )


    #funneling up
    up_sf = [sf for sf in reversed(h_sfs)][1:]
    up_nl = [nl for nl in reversed(h_nl)][1:]
    up_layers = nn.ModuleList()
    for sf, n_layers in reversed(list(zip(up_sf, up_nl))):
        layer = Layer(
            dim=dim,
            shortening_factor=sf,
            dropout=dropout,
            n_heads=num_of_heads,
            block_size=block_size,
            num_blocks=n_layers,
            d_ff=d_ff,
            downflag=False,
            condition_every_n_layers=condition_every_n_layers,
            use_conditioning=use_conditioning,
            rolling_kv_cache=rolling_kv_cache
        )
        up_layers.append(layer)

    return down_layers, center_layer, up_layers

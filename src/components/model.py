import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.components.Attention import MultiHeadAttention, SlidingWindowAttention
from src.components.PerceiverEncoder import get_encoder
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
from src.components.PositionalEncoding import RoPEncoding
from src.config_entities import ModelParams
from src.components.VertexTokenizer import VertexTokenizer
from src.config_entities import ConditioningConfig
from src.components.RollingKV import RollingKVCache

class Meshtron(nn.Module):
    """
    Meshtron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale
    
    An autoregressive transformer model capable of generating high-quality 3D meshes with up to 64k faces
    at 1024-level coordinate resolution. Based on NVIDIA's research paper "Meshtron: High-Fidelity, 
    Artist-Like 3D Mesh Generation at Scale" (2024).
    
    Meshtron addresses the scalability challenges of mesh generation through four key innovations:
    1. Hourglass neural architecture that captures hierarchical mesh structure
    2. Truncated sequence training with sliding window inference  
    3. Global conditioning via cross-attention to point cloud embeddings
    4. Robust sampling with mesh sequence ordering enforcement
    
    The model generates meshes autoregressively by predicting vertex coordinates sequentially,
    with meshes represented as ordered sequences following bottom-to-top, lexicographic ordering.
    The Hourglass architecture allocates more compute to harder-to-predict tokens (typically the 
    last coordinates in each face/vertex group due to vertex sharing patterns).
    
    The model is trained on truncated mesh sequences and uses sliding window inference to generate
    complete meshes. Point cloud conditioning is handled by a jointly-trained Perceiver encoder that
    encodes input point clouds into fixed-size embeddings. Additional conditioning signals include
    target face count and quad face proportion for controlling mesh density and tessellation style.
    
    Key architectural features:
    - Hourglass transformer with shortening factor of 3 (aligned with mesh hierarchy)
    - Cross-attention layers for global conditioning every 4th layer
    - RoPE positional embeddings compatible with rolling KV-cache
    - Mesh sequence ordering enforcement during sampling
    - Support for both triangle and quad-dominant mesh generation
    """
    def __init__(self,
                 *,
                 dim: int,
                 embedding_size: int,
                 n_heads: int,
                 block_size: int,
                 d_ff: int,
                 hierarchy: str,
                 dropout: float,
                 seq_len: int,
                 tokenizer: VertexTokenizer,
                 use_conditioning: bool,
                 condition_every_n_layers:int,
                 use_kv_cache: bool,
                 rolling_max_seq: int,
                 rope_theta: int,
                 conditioning_params: ConditioningConfig,
                 training: bool,
                 window_size: int
                 ):
        """
            Initialize Meshtron Model
            Args:
                dim (int): Primary embedding dimension for the model's hidden states.
                embedding_size (int): number of the rows for quantized coordinate tokens and special tokens (typically 1024 + 3 or 128 + 3).
                n_heads (int): Number of attention heads in the main transformer layers.
                attn_window_size (int): Size of the sliding attention window for local attention mechanisms.
                d_ff (int): Hidden dimension of the feed-forward networks.
                hierarchy (str): Configuration string specifying the Hourglass transformer layer distribution
                                (e.g., "4-8-12" for 4 layers at coord level, 8 at vertex level, 12 at face level).
                dropout (float): Dropout probability applied throughout the network for regularization.
                use_conditioning (bool): Whether to enable point cloud conditioning via cross-attention.
                con_num_latents (int): Number of latent embeddings from the point cloud encoder (typically 1024).
                con_latent_dim (int): Dimensionality of each point cloud latent embedding.
                con_dim_ffn (int): Hidden dimension for feed-forward networks in conditioning/encoder blocks.
                con_num_blocks (int): Number of transformer blocks in the point cloud encoder.
                con_n_attn_heads (int): Number of attention heads in the point cloud encoder.
                con_num_self_attention_blocks (int): Number of self-attention blocks within the encoder.
                condition_every_n_layers (int): Frequency of cross-attention conditioning injection into main layers
                                            (typically every 4th layer following LLaMA-style architecture).
        """
        super().__init__()
        shortening_factor, num_blocks, n_pre_post_blocks = parse_hierarchy(hierarchy=hierarchy)

        funnel_n_layers = 0
        for i in num_blocks[:-1]:
            funnel_n_layers += i 
        total_layers = n_pre_post_blocks + (funnel_n_layers * 2) + num_blocks[1]
        self.rolling_kv_cache = RollingKVCache(total_layers, n_heads, dim // n_heads, rolling_max_seq) if use_kv_cache else None


        self.sf = shortening_factor
        self.n_blocks = num_blocks
        self.embedding = InputEmbedding(embedding_size, dim)
        self.up_sample = LinearUpSample(shortening_factor[0], dim)
        self.pos_emb = RoPEncoding(dim=dim, seq_len=seq_len, theta=rope_theta)
        self.use_conditioning = use_conditioning
        self.tokenizer = tokenizer
        self.point_cloud_conditioning = get_encoder(conditioning_params=conditioning_params)
        self.pre_blocks = nn.ModuleList([
            Transformer(
                dim,
                dropout,
                MultiHeadAttention(dim, n_heads, dropout),
                FeedForwardNetwork(dim, d_ff, dropout, SwiGLU),
                conditioning_flag= use_conditioning and (i % condition_every_n_layers == 0) and i != 0
            )
            for i in range(n_pre_post_blocks)
        ])

        self.down_valley, self.center_layer, self.up_valley = build_hourglass_valley(
            dim=dim,
            num_of_heads=n_heads,
            block_size=block_size,
            h_sfs=self.sf,
            h_nl=self.n_blocks,
            d_ff=d_ff,
            training=training,
            window_size=window_size,
            dropout=dropout,
            pad_token=tokenizer.PAD.item(),
            condition_every_n_layers=condition_every_n_layers,
            use_conditioning=use_conditioning,
            rolling_kv_cache=self.rolling_kv_cache
        )
                                             
        self.post_block = nn.ModuleList([
            Transformer(dim, 
                        dropout, 
                        MultiHeadAttention(dim, n_heads, dropout),
                        FeedForwardNetwork(dim, d_ff, dropout, SwiGLU),
                        conditioning_flag= use_conditioning and (i % condition_every_n_layers == 0) and i != 0
            ) for i in range(n_pre_post_blocks)
        ])

        self.out_proj = ProjectionLayer(dim, embedding_size)
        
    
    def forward(self, data, conditioning_data, face_count, quad_ratio, mask):

        #conditioning tensor
        cond = self.point_cloud_conditioning(conditioning_data, face_count, quad_ratio).to(dtype=torch.float16)
        def run_block(block: Transformer, data):
            conditions = cond if block.conditioning_flag else None
            return block(x = data, conditions=conditions, mask=mask, rolling_kv_cache = self.rolling_kv_cache)
        
        skips = [] #holds skip connection values, used in upsampling

        data = self.embedding(data)
        data = self.pos_emb(data)

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

def get_model(model_params: ModelParams):
    return Meshtron(
            dim = model_params.dim,
            embedding_size= model_params.embedding_size,
            n_heads= model_params.n_heads,
            block_size= model_params.block_size,
            d_ff= model_params.d_ff,
            hierarchy= model_params.hierarchy,
            dropout= model_params.dropout,
            seq_len=model_params.seq_len,
            tokenizer = model_params.tokenizer,
            use_conditioning= model_params.use_conditioning,
            condition_every_n_layers= model_params.condition_every_n_layers,
            use_kv_cache=model_params.use_kv_cache,
            rolling_max_seq=model_params.rolling_max_seq,
            rope_theta=model_params.rope_theta,
            conditioning_params=model_params.conditioning_config,
            training=model_params.training,
            window_size=model_params.window_size,
        )
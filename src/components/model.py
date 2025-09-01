import torch.nn as nn
from src.components.Attention import MultiHeadAttention
from src.components.PerceiverEncoder import ConditioningEncoder
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
        """
            Initialize Meshtron Model
            Args:
                dim (int): Primary embedding dimension for the model's hidden states.
                vocab_size (int): Size of the vocabulary for quantized coordinate tokens (typically 1024).
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

    def forward(self, x, conditioning_data, face_count, quad_ratio, mask):

        def run_block(block):
            cond = self.point_cloud_conditioning(conditioning_data, face_count, quad_ratio) if block.use_conditioning else None
            x = block(x=x, conditions=cond, mask=mask)
            return x
        
        skips = [] #holds skip connection values, used in upsampling
        x = self.embedding(x)

        #conditioning tensor
        cond= None

        # Pre valley block
        for block in self.pre_blocks:
            x = run_block(block)
        skips.append(x) # Appending residuals to be added later

        #Downsampling valley
        for block in self.down_valley:
            x = run_block(block)
            skips.append(x)

        #center layer of valley
        x = self.center_layer(x = x, conditions = cond, mask = mask)

        
        #upsampling valley
        for block, skip in zip(self.up_valley, reversed(skips[1:])):
            x = self.up_sample(x, skip)
            x = run_block(block)

        #upsampling for the last vanilla block
        x = self.up_sample(x, skips[0])

        #last vanilla blocks
        for block in self.post_blocks:
            x = run_block(block)
        
        return self.out_proj(x)
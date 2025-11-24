import torch
import torch.nn as nn
from perceiver_pytorch import Perceiver
    
class ConditioningEncoder(nn.Module):
    """Conditions the meshtron model on 3d point cloud"""
    def __init__(self,
                 *,
                 input_channels: int,          
                 input_axis: int,              
                 num_freq_bands: int,          
                 max_freq: float,              
                 depth: int,                                        
                 num_latents: int,           
                 latent_dim: int,            
                 cross_heads: int,             
                 latent_heads: int,            
                 cross_dim_head: int,         
                 latent_dim_head: int,        
                 num_classes: int,          
                 attn_dropout: float,
                 ff_dropout: float,
                 weight_tie_layers: bool,   
                 fourier_encode_data: bool,  
                 self_per_cross_attn: int,   
                 final_classifier_head:bool,
                 dim_ffn:int,
                 ):
        super().__init__()

        self.model = Perceiver(
            num_freq_bands=num_freq_bands,
            depth=depth,
            max_freq=max_freq,
            input_channels=input_channels,
            input_axis=input_axis,
            num_latents = num_latents,
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            num_classes=num_classes,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            weight_tie_layers=weight_tie_layers,
            fourier_encode_data=fourier_encode_data,
            self_per_cross_attn=self_per_cross_attn,
            final_classifier_head=final_classifier_head
        )
        
        self.face_count_mlp = nn.Sequential(
            nn.Linear(1, dim_ffn),
            nn.ReLU(),
            nn.Linear(dim_ffn, latent_dim)
        )
        self.quad_ratio_mlp = nn.Sequential(
            nn.Linear(1, dim_ffn),
            nn.ReLU(),
            nn.Linear(dim_ffn, latent_dim)
        )
                
    def forward(self, point_input, face_count,
                 quad_ratio):
        
        b = point_input.shape[0]

        #return_embeddings = True - Returns the latents instead of class logits
        latents = self.model(data = point_input, return_embeddings = True) 

        #(B) -> (B, 1)
        face_count = face_count.view(b, -1)
        quad_ratio = quad_ratio.view(b, -1)

        face_count_enc = self.face_count_mlp(face_count)
        quad_ratio_enc = self.quad_ratio_mlp(quad_ratio)

        face_count_enc.unsqueeze_(1)
        quad_ratio_enc.unsqueeze_(1)

        return torch.cat([latents, face_count_enc, quad_ratio_enc], dim=1)
    

from meshtron.model import Meshtron
from meshtron.encoder_conditioning import ConditioningEncoder
from pipeline.config_entities import ModelParams, ConditioningConfig, TrainingConfig
from pipeline.config import ConditioningConfig
from pipeline.utils.common import get_path, get_root_folder
from pathlib import Path
import os

def get_model(model_params: ModelParams):
    return Meshtron(
            dim = model_params.dim,
            embedding_size= model_params.embedding_size,
            n_heads= model_params.n_heads,
            head_dim=model_params.head_dim,
            window_size=model_params.window_size,
            d_ff= model_params.dim_ff,
            hierarchy= model_params.hierarchy,
            dropout= model_params.dropout,
            condition_every_n_layers= model_params.condition_every_n_layers,
            encoder=get_encoder(ConditioningConfig)
        )

def get_encoder(conditioning_params: ConditioningConfig):
    return ConditioningEncoder(
            input_channels=conditioning_params.input_channels,
            input_axis=conditioning_params.input_axis,
            num_freq_bands=conditioning_params.num_freq_bands,
            max_freq=conditioning_params.max_freq,
            depth=conditioning_params.depth,
            num_latents=conditioning_params.num_latents,
            latent_dim=conditioning_params.latent_dim,
            cross_heads=conditioning_params.cross_heads,
            latent_heads=conditioning_params.latent_heads,
            cross_dim_head=conditioning_params.cross_dim_head,
            latent_dim_head=conditioning_params.latent_dim_head,
            num_classes=conditioning_params.num_classes,
            attn_dropout=conditioning_params.attn_dropout,
            ff_dropout=conditioning_params.ff_dropout,
            weight_tie_layers=conditioning_params.weight_tie_layers,
            fourier_encode_data=conditioning_params.fourier_encode_data,
            self_per_cross_attn=conditioning_params.self_per_cross_attn,
            final_classifier_head=conditioning_params.final_classifier_head,
            dim_ffn=conditioning_params.dim_ffn
        )

def get_latest_weights_path(config: TrainingConfig):
    weights_files = list(Path(config.model_folder).glob(config.model_basename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

def get_weights_path(config: TrainingConfig, epoch: str):
    PROJECT_ROOT = get_root_folder()
    os.makedirs(get_path(PROJECT_ROOT, config.model_folder), exist_ok = True)
    return get_path(PROJECT_ROOT, config.model_folder, f"{config.model_basename}_{epoch}.pt")
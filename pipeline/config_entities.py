from dataclasses import dataclass
from meshtron.encoder_conditioning import ConditioningEncoder
@dataclass
class IngestionConfig:
    root: str
    dataset_storage_dir: str
    meshes: str
    dataset_len: int

@dataclass
class DatasetConfig:
    dataset_dir: str
    original_mesh_dir: str
    point_cloud_size: int 
    num_of_bins: int 
    bounding_box_dim: float
    std_points:float
    mean_points:float 
    mean_normals:float
    std_normals:float
    
@dataclass
class ConditioningConfig:
    num_freq_bands: int
    depth: int
    max_freq: float
    input_channels: int
    input_axis: int
    num_latents: int
    latent_dim: int
    cross_heads: int
    latent_heads: int
    cross_dim_head: int
    latent_dim_head: int
    num_classes: int
    attn_dropout: float
    ff_dropout: float
    weight_tie_layers: int
    fourier_encode_data: bool
    self_per_cross_attn: int
    final_classifier_head: bool
    dim_ffn:int

@dataclass
class ModelParams:
    dim: int
    embedding_size: int
    n_heads: int
    head_dim: int
    window_size: int
    dim_ff: int
    hierarchy: str
    dropout: float
    pad_token: int
    condition_every_n_layers: int
    encoder: ConditioningEncoder
    
@dataclass
class TrainingConfig:
    num_epochs: int
    model_folder: str
    model_basename: str
    learning_rate: float
    label_smoothing: float
    preload: str
    val_after_every: int


@dataclass
class DataLoaderConfig:
    train_ratio: float
    batch_size: int
    num_workers: int
    shuffle: bool
    pin_memory: bool
    persistent_workers:bool

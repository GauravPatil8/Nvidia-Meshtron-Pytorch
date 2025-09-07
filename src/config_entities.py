from dataclasses import dataclass

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
    
@dataclass
class ModelParams:
    dim: int
    embedding_size: int
    n_heads: int
    attn_window_size: int
    d_ff: int
    hierarchy: str
    dropout: float
    use_conditioning: bool
    con_num_latents:int
    con_latent_dim:int
    con_dim_ffn:int
    con_num_blocks:int
    con_n_attn_heads:int
    con_num_self_attention_blocks: int
    condition_every_n_layers: int
    
@dataclass
class TrainingConfig:
    num_epochs: int
    model_folder: str
    model_basename: str
    learning_rate: float
    label_smoothing: float
    preload: str


@dataclass
class DataLoaderConfig:
    train_ratio: float
    batch_size: int
    num_workers: int
    shuffle: bool
    pin_memory: bool
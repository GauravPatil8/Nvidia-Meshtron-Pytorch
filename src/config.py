import os
from pathlib import Path
from src.utils.common import get_path, get_root_folder
from src.config_entities import IngestionConfig, ModelParams, TrainingConfig, DatasetConfig,DataLoaderConfig

class ConfigurationManager:

    @staticmethod
    def ingestion_config():
        PROJECT_ROOT = get_root_folder()
        return IngestionConfig(
            root = get_path(PROJECT_ROOT, 'artifacts'),
            dataset_storage_dir = get_path(PROJECT_ROOT, 'artifacts', 'dataset'),
            meshes = get_path(PROJECT_ROOT, 'mesh'),
            dataset_len = 50
        )
    
    @staticmethod
    def trainig_config():
        PROJECT_ROOT = get_root_folder()
        return TrainingConfig(
            learning_rate=0.01,
            label_smoothing= 0.1,
            model_folder=get_path(PROJECT_ROOT, "artifacts", "model"),
            model_basename="meshtron",
            preload="latest",
        )
    
    @staticmethod
    def model_params():
        return ModelParams(
            dim = 12000,
            embedding_size = 130, # 128 + 2,
            n_heads = 64,
            attn_window_size=3,
            d_ff=512,
            hierarchy="1@2 2@4 4@2 2@4 1@2",
            dropout = 0.3,
            use_conditioning = True,
            con_num_latents= 1024,
            con_latent_dim=512,
            con_dim_ffn=512,
            con_num_blocks=8,
            con_n_attn_heads=64,
            con_num_self_attention_blocks= 8,
            condition_every_n_layers= 4         
        )

    @staticmethod
    def dataset_config():
        PROJECT_ROOT = get_root_folder()
        return DatasetConfig(
            dataset_dir=get_path(PROJECT_ROOT, 'artifacts', 'dataset'),
            original_mesh_dir=get_path(PROJECT_ROOT, 'mesh'),
            point_cloud_size=8192,
            num_of_bins=128,
            bounding_box_dim=1.0
        )
    
    @staticmethod
    def dataloader_config():
        return DataLoaderConfig(
            train_ratio=0.9,
            batch_size=32,
            num_workers=2,
            shuffle=True,
            pin_memory=False
        )
        
def get_latest_weights_path(config: TrainingConfig):
    weights_files = list(Path(config.model_folder).glob(config.model_basename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

def get_weights_path(config: TrainingConfig, epoch: str):
    PROJECT_ROOT = get_root_folder()
    return get_path(PROJECT_ROOT, config.model_folder, f"{config.model_basename}_{epoch}.pt")
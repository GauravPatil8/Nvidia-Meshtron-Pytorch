import os
from pathlib import Path
from src.utils.data import get_max_seq_len
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
    def training_config():
        PROJECT_ROOT = get_root_folder()
        return TrainingConfig(
            num_epochs=1,
            learning_rate=0.01,
            label_smoothing= 0.1,
            model_folder=get_path(PROJECT_ROOT, "artifacts", "model"),
            model_basename="meshtron",
            preload="latest",
        )
    
    @staticmethod
    def model_params():
        PROJECT_ROOT = get_root_folder()
        return ModelParams(
            dim = 512,
            embedding_size = 131, # 0-127(bins) + 128-130 special tokens,
            n_heads = 64,
            block_size=3,
            d_ff=512,
            hierarchy="4@1 8@3 12@9 8@3 4@1",
            dropout = 0.3,
            seq_len = get_max_seq_len(get_path(PROJECT_ROOT, 'mesh')),
            tokenizer=None,
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
            bounding_box_dim=1.0,
            std_points=0.1,
            mean_points=0.0
        )
    
    @staticmethod
    def dataloader_config():
        return DataLoaderConfig(
            train_ratio=0.9,
            batch_size=10,
            num_workers=0,
            shuffle=True,
            pin_memory=False,
            persistent_workers=False
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
import os
from pathlib import Path
from pipeline.utils.data import get_max_seq_len
from pipeline.utils.common import get_path, get_root_folder
from pipeline.config_entities import (
    IngestionConfig, 
    ModelParams, 
    TrainingConfig, 
    DatasetConfig,
    DataLoaderConfig, 
    ConditioningConfig
)

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
            num_epochs=2,
            learning_rate=0.01,
            label_smoothing= 0.,
            model_folder=get_path(PROJECT_ROOT, "artifacts", "model"),
            model_basename="meshtron",
            preload="latest",
            val_after_every=500,
        )
    
    @staticmethod
    def model_params():
        #configured according to Meshtron-small
        PROJECT_ROOT = get_root_folder()
        return ModelParams(
            dim = 1024,
            embedding_size = 131, # 0-127(bins) + 128-130 special tokens,
            n_heads = 64,
            head_dim=64,
            window_size=3,
            dim_ff=2816,
            hierarchy="4@1 8@3 12@9 8@3 4@1",
            dropout = 0.,
            condition_every_n_layers= 4,   
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
            std_points=0.01,
            mean_points=0.0,
            mean_normals=0.0,
            std_normals=0.03
        )
    
    @staticmethod
    def dataloader_config():
        return DataLoaderConfig(
            train_ratio=1.0,
            batch_size=2,
            num_workers=2,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )
    @staticmethod 
    def conditioning_config():
        return ConditioningConfig(
            num_freq_bands= 6,
            depth = 8,
            max_freq = 10.,
            input_channels=6,
            input_axis= 1,
            num_latents = 1024,
            latent_dim = 1024,
            cross_heads = 1,
            latent_heads = 16,
            cross_dim_head = 64,
            latent_dim_head = 64,
            num_classes = 1,
            attn_dropout = 0.1,
            ff_dropout= 0.0,
            weight_tie_layers = 6,
            fourier_encode_data = True,
            self_per_cross_attn = 2,
            final_classifier_head = False,
            dim_ffn = 2816
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
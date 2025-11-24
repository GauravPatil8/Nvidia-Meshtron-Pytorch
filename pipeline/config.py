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
            dataset_len = 100
        )
    
    @staticmethod
    def training_config():
        PROJECT_ROOT = get_root_folder()
        return TrainingConfig(
            num_epochs=1000,
            learning_rate=1e-4,
            label_smoothing= 0.1,
            model_folder=get_path(PROJECT_ROOT, "artifacts", "models"),
            model_basename="meshtron",
            preload="latest",
            val_after_every=2000,
        )
    
    @staticmethod
    def model_params():
        #Custom configuration to manage low diversity in primitive dataset
        return ModelParams(
            dim = 512,
            embedding_size = 131,
            n_heads = 16,
            head_dim = 32,
            window_size = 256,
            dim_ff = 1536,
            shortening_factor= 3,
            num_blocks_per_layer=[4,8,12],
            ff_dropout = 0.0,
            attn_dropout = 0.0,
            pad_token = 0,
            condition_every_n_layers = 4,
            encoder = None
        )

    @staticmethod
    def dataset_config():
        PROJECT_ROOT = get_root_folder()
        return DatasetConfig(
            dataset_dir=get_path(PROJECT_ROOT, 'artifacts', 'dataset'),
            original_mesh_dir=get_path(PROJECT_ROOT, 'mesh'),
            point_cloud_size=8192//2,
            num_of_bins=128,
            std_points=0.01,
            mean_points=0.0,
            mean_normals=0.0,
            std_normals=0.03
        )
    
    @staticmethod
    def dataloader_config():
        return DataLoaderConfig(
            train_ratio=0.9,
            batch_size=12,
            num_workers=2,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )
    
    @staticmethod
    def conditioning_config():
        return ConditioningConfig(
            num_freq_bands = 4,
            depth = 4,
            max_freq = 6.0,
            input_channels = 6,
            input_axis = 1,
            num_latents = 256,
            latent_dim = 512,
            cross_heads = 8,
            latent_heads = 8,
            cross_dim_head = 32,
            latent_dim_head = 32,
            num_classes = 1,
            attn_dropout = 0.0,
            ff_dropout = 0.0,
            weight_tie_layers = 2,
            fourier_encode_data = True,
            self_per_cross_attn = 1,
            final_classifier_head = False,
            dim_ffn = 1024
    )
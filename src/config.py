import os
from utils.common import get_path, get_root_folder
from config_entities import IngestionConfig, Params

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
    def model_params():
        return Params(
            dim = 12500,

        )
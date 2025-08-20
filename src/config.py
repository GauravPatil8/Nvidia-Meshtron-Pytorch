import os
from utils.common import get_path, get_root_folder
from config_entities import IngestionConfig

class ConfigurationManager:

    @staticmethod
    def ingestion_config():
        PROJECT_ROOT = get_root_folder()
        return IngestionConfig(
            root = get_path(PROJECT_ROOT, 'artifacts'),
            dataset = get_path(PROJECT_ROOT, 'artifacts', 'dataset'),
            meshes = get_path(PROJECT_ROOT, 'mesh')
        )

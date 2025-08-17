import os
from utils.common import get_path
from config_entities import IngestionConfig

class ConfigurationManager:

    @staticmethod
    def ingestion_config():
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        return IngestionConfig(
            root = get_path(PROJECT_ROOT, 'artifacts'),
            dataset = get_path(PROJECT_ROOT, 'artifacts', 'dataset'),
            meshes = get_path(PROJECT_ROOT, 'mesh')
        )

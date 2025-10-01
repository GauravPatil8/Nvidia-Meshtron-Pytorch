import torch
from src.stages.ingestion import Ingestion
from src.stages.training import Trainer
from src.config import ConfigurationManager

class Pipeline:
    
    @staticmethod
    def run():
        torch.cuda.empty_cache()

        ingestion_stage = Ingestion(ConfigurationManager.ingestion_config())
        ingestion_stage.run()
            
        training_stage = Trainer(ConfigurationManager.training_config(),
                                 ConfigurationManager.model_params(),
                                 ConfigurationManager.dataset_config(),
                                 ConfigurationManager.dataloader_config())
        training_stage.run()



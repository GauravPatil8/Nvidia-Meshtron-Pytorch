import torch
from pipeline.stages.ingestion import Ingestion
from pipeline.stages.training import Trainer
from pipeline.config import ConfigurationManager

class PipelineRunner:
    
    @staticmethod
    def run():
        torch.cuda.empty_cache()

        ingestion_stage = Ingestion(ConfigurationManager.ingestion_config())
        ingestion_stage.run()
            
        training_stage = Trainer(ConfigurationManager.ingestion_config().dataset_len,
                                 ConfigurationManager.training_config(),
                                 ConfigurationManager.model_params(),
                                 ConfigurationManager.dataset_config(),
                                 ConfigurationManager.dataloader_config())
        training_stage.run()



import torch
from src.stages.ingestion import Ingestion
from src.stages.training import Trainer
from src.config import ConfigurationManager

class Pipeline:
    def __init__(self):
        self._stages = list()

        # Ingestion stage
        ingestion_stage = Ingestion(ConfigurationManager.ingestion_config())
        self._stages.append(ingestion_stage)

        #Training stage
        training_stage = Trainer(ConfigurationManager.training_config(),
                                 ConfigurationManager.model_params(),
                                 ConfigurationManager.dataset_config(),
                                 ConfigurationManager.dataloader_config())
        self._stages.append(training_stage)

    def run(self):
        torch.cuda.empty_cache()
        for stage in self._stages:
            stage.run()
            



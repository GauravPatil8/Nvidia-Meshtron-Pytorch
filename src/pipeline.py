import os
from stages.ingestion import Ingestion
from stages.training import Trainer
from config import ConfigurationManager

class Pipeline:
    def __init__(self):
        self._stages = []

        #Ingestion stage
        ingestion_stage = Ingestion(ConfigurationManager.ingestion_config())
        self._stages.append(ingestion_stage)

        #Training stage
        training_stage = Trainer(ConfigurationManager.trainig_config(),
                                 ConfigurationManager.model_params(),
                                 ConfigurationManager.dataloader_config(),
                                 ConfigurationManager.dataloader_config())
        self._stages.append(training_stage)

    def run(self):
        for stage in self._stages:
            stage.run()
            



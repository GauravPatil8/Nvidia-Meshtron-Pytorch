import os
from stages.ingestion import Ingestion
from config import ConfigurationManager

class Pipeline:
    def __init__(self):
        self._stages = []

        #Ingestion stage
        ing_config = ConfigurationManager.ingestion_config()
        ingestion_stage = Ingestion(len(os.listdir(ing_config.meshes)) * 1000, ing_config.meshes, ing_config.dataset)
        self._stages.append(ingestion_stage)

    def run(self):
        for stage in self._stages:
            stage.run()
            
if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.run()


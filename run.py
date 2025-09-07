from src.pipeline import Pipeline
from src.utils.common import logger_init

if __name__ == "__main__":
    logger_init()
    pipeline = Pipeline()
    pipeline.run()
import os
import sys
from pipeline.pipeline_runner import PipelineRunner

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    PipelineRunner.run()
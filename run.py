import os
import sys
from src.pipeline import Pipeline

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    Pipeline.run()
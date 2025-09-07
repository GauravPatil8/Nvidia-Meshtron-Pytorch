import os
import logging

def get_path(*args) -> str:
    "Joins and returns path (str)"
    return os.path.join(*args)

def get_root_folder():
    "returns the project's root folder path"
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def logger_init():
    os.makedirs(R"src/logs/", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(R"src/logs/info.log"),
            logging.StreamHandler()
        ]
    )
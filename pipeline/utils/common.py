import os
import logging

def get_path(*args) -> str:
    "Joins and returns path (str)"
    return os.path.join(*args)

def get_root_folder():
    "returns the project's root folder path"
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def logger_init(name="meshtron_logger"):
   
    os.makedirs("pipeline/logs/", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevent bubbling to root logger

    # Prevent adding duplicate handlers if re-initialized
    if logger.handlers:
        return logger

    # File handler only
    fh = logging.FileHandler("pipeline/logs/info.log")
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    return logger

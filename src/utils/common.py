import os
import logging

def get_path(*args) -> str:
    "Joins and returns path (str)"
    return os.path.join(*args)

def get_root_folder():
    "returns the project's root folder path"
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def logger_init(name="meshtron_logger"):
   
    os.makedirs("src/logs/", exist_ok=True)

    # Create a dedicated logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid propagating messages to root (prevents external libs logging here)
    logger.propagate = False  

    # File handler
    fh = logging.FileHandler("src/logs/info.log")
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers (avoid duplicates if re-initialized)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

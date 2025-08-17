import os

def get_path(*args) -> str:
    "Joins and returns path (str)"
    return os.path.join(*args)

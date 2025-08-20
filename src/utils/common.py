import os

def get_path(*args) -> str:
    "Joins and returns path (str)"
    return os.path.join(*args)

def get_root_folder():
    "returns the project's root folder path"
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

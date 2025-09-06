from dataclasses import dataclass

@dataclass
class IngestionConfig:
    root: str
    dataset_storage_dir: str
    meshes: str
    dataset_len: int

@dataclass
class Params:
    dim: int

class TrainingConfig:
    params : Params
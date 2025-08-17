from dataclasses import dataclass

@dataclass
class IngestionConfig:
    root: str
    dataset: str
    meshes: str
    len_dataset: int

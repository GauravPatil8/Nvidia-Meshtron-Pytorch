import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import trimesh
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, random_split
from src.utils.common import get_path
from src.components.VertexTokenizer import VertexTokenizer
from src.utils.data import get_mesh_stats, get_max_seq_len, normalize_mesh_to_bbox
from src.config_entities import DatasetConfig, DataLoaderConfig

class PrimitiveDataset(Dataset):
    def __init__(self,
                 *,
                  dataset_dir: str, 
                  original_mesh_dir: str,
                  tokenizer: VertexTokenizer, 
                  point_cloud_size: int = 2048, 
                  num_of_bins: int = 1024, 
                  bounding_box_dim: float = 1.0
                  ):
        """
            Dataset class to handle mesh dataset.
            Parameters:
                dataset_dir = location of stored data.
                point_cloud_size = number of points to sample on the mesh
                num_of_bins = number of bins to map the values
                bounding_box_dim = length of ont side of box
        """
        if os.path.exists(dataset_dir):
            self.data_dir = dataset_dir
        else:
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        self.max_seq_len = get_max_seq_len(original_mesh_dir)
        self.num_points = point_cloud_size
        self.num_of_bins = num_of_bins
        self.bounding_box_dim = bounding_box_dim
        self.files = [get_path(root, file) for root, _ , files in os.walk(dataset_dir) for file in files]
        self.tokenizer = tokenizer
        self.EOS = self.tokenizer.EOS
        self.SOS = self.tokenizer.SOS
        self.PAD = self.tokenizer.PAD
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):

        #get the stats before triangulation and normalization
        face_count, quad_ratio = get_mesh_stats(self.files[index])

        mesh = trimesh.load_mesh(self.files[index])# returns triangulated mesh by default
        
        vertices = normalize_mesh_to_bbox(self.files[index], self.bounding_box_dim)

        mesh.vertices = vertices

        #sampling points on the surface of the bounded mesh (N, 3)
        point_cloud = mesh.sample(self.num_points)

        #decoder input
        dec_input = self.tokenizer.encode(mesh, vertices)

        print(len(dec_input))
        #add special tokens
        num_dec_tokens = 0
        # if self.max_seq_len != len(dec_input):
        #     print(f"file {self.files[index]}")
        num_dec_tokens = self.max_seq_len - len(dec_input) 

        if num_dec_tokens < 0:
            print(f"[ERROR] File: {self.files[index]}")
            print(f"Max seq len allowed: {self.max_seq_len}")
            print(f"Got length: {len(dec_input)}") 
            raise ValueError("Sentence is too long")
        
        decoder_input = torch.cat(
            [
                torch.tensor([self.SOS] * 9, dtype=torch.int32), #for preserving hourglass structure
                dec_input,
                torch.tensor([self.PAD] * num_dec_tokens, dtype=torch.int32)
            ],
            dim=0
        )

        target = torch.cat(
            [
                dec_input,
                torch.tensor([self.EOS] * 9, dtype=torch.int32), #for preserving hourglass structure
                torch.tensor([self.PAD] * num_dec_tokens, dtype=torch.int32)
            ],
            dim=0
        )
        return {
            "decoder_input":decoder_input,
            "decoder_mask":(decoder_input != self.PAD).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),,
            "target":target,
            "point_cloud":torch.from_numpy(point_cloud).to(dtype=torch.float32),
            "quad_ratio":torch.tensor(quad_ratio, dtype=torch.float32),
            "face_count":torch.tensor(face_count, dtype=torch.float32),
        }


def causal_mask(size):
    mask = torch.tril(torch.ones((1, size, size))).type(torch.int)
    return mask == 1


def get_dataloaders(dataset_config: DatasetConfig, loader_config: DataLoaderConfig):
    """Returns Train and test split dataloaders and VertexTokenizer"""
    vertex_tokenizer = VertexTokenizer(dataset_config.num_of_bins, dataset_config.bounding_box_dim)
    dataset = PrimitiveDataset(
        dataset_dir=dataset_config.dataset_dir,
        original_mesh_dir=dataset_config.original_mesh_dir,
        tokenizer=vertex_tokenizer,
        point_cloud_size=dataset_config.point_cloud_size,
        num_of_bins=dataset_config.num_of_bins,
        bounding_box_dim=dataset_config.bounding_box_dim
    )

    dataset_size = len(dataset)
    train_size = int(dataset_size * loader_config.train_ratio)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle,
        num_workers=loader_config.num_workers,
        pin_memory=loader_config.pin_memory
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle,
        num_workers=loader_config.num_workers,
        pin_memory=loader_config.pin_memory
    )

    return train_loader, test_loader, vertex_tokenizer

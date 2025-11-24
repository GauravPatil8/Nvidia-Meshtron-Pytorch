import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import trimesh
from torch.utils.data import Dataset, DataLoader, random_split
from pipeline.utils.common import get_path
from meshtron.VertexTokenizer import VertexTokenizer
from pipeline.utils.data import get_mesh_stats, get_max_seq_len, normalize_verts_to_box, add_gaussian_noise, set_zero_vector
from pipeline.config_entities import DatasetConfig, DataLoaderConfig

class PrimitiveDataset(Dataset):
    def __init__(self,
                 *,
                  dataset_dir: str, 
                  original_mesh_dir: str,
                  tokenizer: VertexTokenizer, 
                  point_cloud_size: int = 2048, 
                  num_of_bins: int = 1024,
                  std_points:float,
                  mean_points:float,
                  mean_normals:float,
                  std_normals:float
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
        self.std_points = std_points
        self.mean_points = mean_points
        self.mean_normals = mean_normals
        self.std_normals = std_normals
        self.max_seq_len = get_max_seq_len(original_mesh_dir)
        self.num_points = point_cloud_size
        self.num_of_bins = num_of_bins
        self.bounding_box_dim = 1.0
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
        
        vertices = normalize_verts_to_box(self.files[index])

        mesh.vertices = vertices

        #sampling points on the surface of the bounded mesh (N, 3)
        point_cloud, face_indices = trimesh.sample.sample_surface(mesh, self.num_points)

        #point cloud & point normals
        point_cloud = torch.from_numpy(point_cloud).to(dtype=torch.float32)
        point_normals = torch.from_numpy(mesh.face_normals[face_indices]).to(dtype=torch.float32)

        # augmentation
        point_cloud = add_gaussian_noise(point_cloud, mean=self.mean_points, std=self.std_points) #according to paper: mean = 0.0, std = 0.01
        point_normals = add_gaussian_noise(point_normals, mean=self.mean_normals, std=self.std_normals)
        point_normals = set_zero_vector(points=point_normals, rate=0.3, size=point_normals.shape[1])

        points = torch.cat((point_cloud, point_normals), dim=1)

        #decoder input
        dec_input = self.tokenizer.encode(mesh, vertices)

        #add special tokens
        num_dec_tokens = 0
        
        num_dec_tokens = self.max_seq_len - len(dec_input) - 9

        if num_dec_tokens < 0:
            print(f"[ERROR] File: {self.files[index]}")
            print(f"Max seq len allowed: {self.max_seq_len}")
            print(f"Got length: {len(dec_input)}") 
            raise ValueError("Sentence is too long")
        
        decoder_input = torch.cat(
            [
                torch.tensor([self.SOS] * 9, dtype=torch.int64), #for preserving hourglass structure
                dec_input,
                torch.tensor([self.PAD] * num_dec_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        target = torch.cat(
            [
                dec_input,
                torch.tensor([self.EOS] * 9, dtype=torch.int64), #for preserving hourglass structure
                torch.tensor([self.PAD] * num_dec_tokens, dtype=torch.int64)
            ],
            dim=0
        )
        return {
            "decoder_input":decoder_input,
            # "decoder_mask":(decoder_input != self.PAD).unsqueeze(0).int() & causal_mask(decoder_input.size(0)).to(dtype=torch.int64), # (seq_len, 1) & (1, seq_len, seq_len)
            "target":target.to(dtype=torch.int64),
            "point_cloud":points.to(dtype=torch.float32),
            "quad_ratio":torch.tensor(quad_ratio, dtype=torch.float32),
            "face_count":torch.tensor(face_count, dtype=torch.float32),
        }

def causal_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze_(0)


def get_dataloaders(dataset_config: DatasetConfig, loader_config: DataLoaderConfig):
    """Returns Train and test split dataloaders and VertexTokenizer"""
    vertex_tokenizer = VertexTokenizer(dataset_config.num_of_bins)
    dataset = PrimitiveDataset(
        dataset_dir=dataset_config.dataset_dir,
        original_mesh_dir=dataset_config.original_mesh_dir,
        tokenizer=vertex_tokenizer,
        point_cloud_size=dataset_config.point_cloud_size,
        num_of_bins=dataset_config.num_of_bins,
        std_points = dataset_config.std_points,
        mean_points = dataset_config.mean_points,
        mean_normals=dataset_config.mean_normals,
        std_normals=dataset_config.std_normals
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
        pin_memory=loader_config.pin_memory,
        persistent_workers=loader_config.persistent_workers
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle,
        num_workers=loader_config.num_workers,
        pin_memory=loader_config.pin_memory,
        persistent_workers=loader_config.persistent_workers
    )

    return train_loader, test_loader, vertex_tokenizer

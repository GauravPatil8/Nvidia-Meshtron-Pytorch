import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import trimesh
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset
from src.utils.common import get_path
from src.components.VertexTokenizer import VertexTokenizer
from src.utils.data import get_mesh_stats, get_max_seq_len, normalize_mesh_to_bbox

@dataclass
class MeshData:
    decoder_input: torch.Tensor
    decoder_mask: torch.Tensor
    target: torch.Tensor
    point_cloud: torch.Tensor
    quad_ratio: float
    face_count: int

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

        #add special tokens
        num_dec_tokens = self.max_seq_len - len(dec_input) - 9

        if num_dec_tokens < 0:
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
        return MeshData(
            decoder_input=decoder_input,
            decoder_mask=(decoder_input != self.PAD).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),,
            target=target,
            point_cloud=point_cloud,
            quad_ratio=quad_ratio,
            face_count=face_count
        )

def causal_mask(size):
    mask = torch.tril(torch.ones((1, size, size))).type(torch.int)
    return mask == 1
    
# if __name__ == '__main__':

#     dataset = PrimitiveDataset(
#         dataset_dir=R"C:\Padhai\implementation_research_papers\meshtron\artifacts\dataset",
#         original_mesh_dir=R"C:\Padhai\implementation_research_papers\meshtron\mesh",
#         tokenizer=VertexTokenizer(1024, 1.0),
#         point_cloud_size= 2048,
#         num_of_bins=1024,
#         bounding_box_dim=1.0,
#     )

#     data1 = dataset[13]
#     print("-"*100)

#     print(data1)
#     print("-"*100)

#     print(f"Quad ratio : {data1.quad_ratio}")
#     print(f"Face count: {data1.face_count}")
#     print("-"*100)

#     print(f"Point cloud : {data1.point_cloud[:10, :]}")
#     print("-"*100)
#     print(f"Decoder input : {data1.decoder_input[10:90]}")
#     print("-"*100)
#     print(f"Decoder mask size: {data1.decoder_mask.size()}")
#     print("-"*100)
#     print(f"Decoder Target size: {data1.target.size()}")
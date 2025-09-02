import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import numpy as np
from src.utils.common import get_path
from torch.utils.data import Dataset
from src.components.VertexTokenizer import VertexTokenizer
from src.utils.data import load_obj, get_mesh_stats, extract_faces_bot_top, lex_sort_verts, get_vertices
from dataclasses import dataclass

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
                  seq_len: int, 
                  tokenizer: VertexTokenizer, 
                  num_points: int = 2048, 
                  num_of_bins: int = 1024, 
                  bounding_box_dim: float = 1.0
                  ):
        """
            Dataset class to handle mesh dataset.
            Parameters:
                dataset_dir = location of stored data.
                num_points = number of points to sample on the mesh
                num_of_bins = number of bins to map the values
                bounding_box_dim = length of ont side of box
        """
        if os.path.exists(dataset_dir):
            self.data_dir = dataset_dir
        else:
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        self.seq_len = seq_len
        self.num_points = num_points
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

        mesh = load_obj(self.files[index]) # returns triangulated mesh by default

        

        face_count, quad_ratio = get_mesh_stats(self.files[index])

        #sampling points on the surface of the bounded mesh (N, 3)
        point_cloud = mesh.sample(self.num_points)


        face_list = extract_faces_bot_top(mesh)
        vertices = torch.tensor(get_vertices(self.files[index]))
        vertices = vertices[:, [2,1,0]]

        sorted_faces_verts = torch.tensor([lex_sort_verts(face, vertices) for face in face_list])

        # Flatten the (N, 3, 3) list to (N*9)
        sequence = torch.flatten(sorted_faces_verts)

        #decoder input
        dec_input = self.tokenizer.encode(sequence)

        #add special tokens
        num_dec_tokens = self.seq_len - len(dec_input) - 1

        if num_dec_tokens < 0:
            raise ValueError("Sentence is too long")
        
        decoder_input = torch.cat(
            [
                self.SOS,
                dec_input,
                torch.tensor([self.PAD] * num_dec_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        target = torch.cat(
            [
                dec_input,
                self.EOS,
                torch.tensor([self.PAD] * num_dec_tokens, dtype=torch.int64)
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
    

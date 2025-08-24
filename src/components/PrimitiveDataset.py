import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import numpy as np
from src.utils.common import get_path
from torch.utils.data import Dataset
from src.components.VertexTokenizer import VertexTokenizer
from src.utils.data import load_obj, normalize_mesh_to_bbox
from dataclasses import dataclass

@dataclass
class MeshData:
    decoder_input: torch.Tensor
    decoder_mask: torch.Tensor
    target: torch.Tensor

class PrimitiveDataset(Dataset):
    def __init__(self, dataset_dir: str, seq_len: int, tokenizer: VertexTokenizer, num_points: int = 2048, num_of_bins: int = 1024, bounding_box_dim: float = 1.0):
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

        mesh = load_obj(self.files[index])

        # normalize and centralize the mesh
        bounded_mesh = normalize_mesh_to_bbox(mesh, self.bounding_box_dim)

        #sampling points on the surface of the bounded mesh (N, 3)
        point_cloud = bounded_mesh.sample(self.num_points)

        #decoder input
        dec_input = self.tokenizer.encode(np.unique(bounded_mesh.vertices, axis=0))

        #encoder input
        enc_input = self.tokenizer.encode(point_cloud)
        
        #add special tokens
        num_enc_tokens = self.seq_len - len(enc_input) - 2
        num_dec_tokens = self.seq_len - len(dec_input) - 1

        if num_dec_tokens < 0 or num_enc_tokens < 0:
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
            target=target
        )

def causal_mask(size):
    mask = torch.tril(torch.ones((1, size, size))).type(torch.int)
    return mask == 1
    

if __name__ == '__main__':
    
    dataset = PrimitiveDataset(R"C:\Padhai\implementation_research_papers\meshtron\artifacts\dataset", num_points=2048, seq_len=
                               6200)

    # Test __len__
    print("Dataset size:", len(dataset))

    # Test one item
    data = dataset[12]

    print("encoder_input: ",data.encoder_input)
    print("decoder_input: ",data.decoder_input)
    # Check shape of points
    print("encoder_input_shape: ",data.encoder_input.shape)
    print("Decoder_input_shape: ", data.decoder_input.shape)

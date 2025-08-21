import torch
import numpy as np

class VertexTokenizer:

    def __init__(self, bins: int, box_dim: float = 1.0):
        "Quantize and add special tokens"
        self.box_dim = box_dim
        self.bins = bins

        #Special tokens
        self.SOS = torch.tensor([bins], dtype=torch.int64)
        self.EOS = torch.tensor([bins+1], dtype=torch.int64)
        self.PAD = torch.tensor([bins+2], dtype=torch.int64) 

        self.vocab_size = bins + 3

    def quantize(self, points: np.array):
        "converts float values to discrete int bins"
        return torch.tensor(np.clip(np.floor((points + (self.box_dim / 2)) * (self.bins / self.box_dim)), 0, self.bins - 1)).to(dtype=torch.int64)

    def dequantize(self, tokens: torch.Tensor):
        "converts int bins to float values"
        return (tokens.float() / (self.bins - 1)) * self.box_dim - (self.box_dim / 2)
    
    def encode(self, points: np.array):

        #Rearrange xyz to yzx as mentioned in the paper
        points = points[:, [1,2,0]]

        #lexsort y->z->x
        sorted_idx = np.lexsort([points[:, 2], points[:, 1], points[:,0]])
        points = points[sorted_idx]

        #flatten the (N,3) array to (n*3) array
        points = points.flatten()

        tokens = self.quantize(points)

        return tokens
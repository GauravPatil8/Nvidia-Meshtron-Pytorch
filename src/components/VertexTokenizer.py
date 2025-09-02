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

    def quantize(self, sequence: np.array):
        "converts float values to discrete int bins"
        return torch.tensor(np.clip(np.floor((sequence + (self.box_dim / 2)) * (self.bins / self.box_dim)), 0, self.bins - 1)).to(dtype=torch.int64)

    def dequantize(self, tokens: torch.Tensor):
        "converts int bins to float values"
        return (tokens.float() / (self.bins - 1)) * self.box_dim - (self.box_dim / 2)
    
    def encode(self, sequence: torch.Tensor):

        tokens = self.quantize(sequence)

        return tokens
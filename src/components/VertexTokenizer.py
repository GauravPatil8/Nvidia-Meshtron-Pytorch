import torch
import trimesh
import numpy as np
from src.utils.data import extract_faces_bot_top, get_vertices, lex_sort_verts, normalize_mesh_to_bbox
class VertexTokenizer:

    def __init__(self, bins: int, box_dim: float = 1.0):
        "Quantize and add special tokens"
        self.box_dim = box_dim
        self.bins = bins

        #Special tokens
        self.SOS = torch.tensor([bins], dtype=torch.int64)
        self.EOS = torch.tensor([bins+1], dtype=torch.int64)
        self.PAD = torch.tensor([bins+2], dtype=torch.int64) 

        self.vocab_size = bins + 3 # add 3 for special tokens

    def quantize(self, sequence: torch.Tensor):
        "converts float values to discrete int bins"
        return (torch.clamp(torch.floor((sequence + (self.box_dim / 2)) * (self.bins / self.box_dim)), 0, self.bins - 1)).to(dtype=torch.int32)

    def dequantize(self, tokens: torch.Tensor):
        "converts int bins to float values"
        return (tokens.float() / (self.bins - 1)) * self.box_dim - (self.box_dim / 2)
    
    def encode(self,mesh: trimesh.Trimesh, vertices: torch.Tensor):
       
        face_list = extract_faces_bot_top(mesh)

        #arrange vertices as x,y,z -> z,y,x. z represents vertical axis.
        vertices = vertices[:, [2,1,0]]

        sorted_faces_verts = torch.from_numpy(np.array([lex_sort_verts(face, vertices) for face in face_list]))

        # Flatten the (N, 3, 3) list to (N*9)
        sequence = torch.flatten(sorted_faces_verts)

        sequence = self.quantize(sequence)

        return sequence
    
    def decode(self):
        pass
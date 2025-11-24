import torch
import trimesh
import numpy as np
from pipeline.utils.data import extract_faces_bot_top, get_vertices, lex_sort_verts, normalize_verts_to_box

class VertexTokenizer:

    def __init__(self, bins: int):
        "Quantize and add special tokens"
        self.box_dim = 1.0
        self.bins = bins

        #Special tokens
        self.SOS = torch.tensor([bins], dtype=torch.int64)
        self.EOS = torch.tensor([bins+1], dtype=torch.int64)
        self.PAD = torch.tensor([bins+2], dtype=torch.int64) 

        self.vocab_size = bins + 3 # add 3 for special tokens

    def quantize(self, sequence: torch.Tensor):
        "converts float values to discrete int bins"
        return (torch.clamp(torch.floor((sequence + (self.box_dim / 2)) * (self.bins / self.box_dim)), 0, self.bins - 1)).to(dtype=torch.int64)

    def dequantize(self, tokens: torch.Tensor):
        "converts integer bins to float values"
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
    
    def decode(self, x:  torch.Tensor):
        """Converts integer tokens to corresponding float coordinates"""

        coordinates = self.dequantize(x)

        #Convert N*3 -> (N,3)
        points = coordinates.view([-1, 3])

        #convert Z Y X -> X Y Z
        points = points[:, [2,1,0]]

        return points
import os
import torch
import trimesh
import numpy as np

class MeshTokenizer:

    def __init__(self, bins: int):
        "Quantize and add special tokens"
        self.box_dim = 1.0
        self.bins = bins

        #Special tokens
        self.SOS = torch.tensor([bins], dtype=torch.int64)
        self.EOS = torch.tensor([bins+1], dtype=torch.int64)
        self.PAD = torch.tensor([bins+2], dtype=torch.int64) 

        self.vocab_size = bins + 3 # add 3 for special tokens

    
    def _extract_faces_bot_top(self, mesh: trimesh.Trimesh):
        "Returns list of faces arranged from bottom to top"

        faces = mesh.faces
        vertices = mesh.vertices

        face_data = []
        for face in faces:
            centroid = np.mean([vertices[i][2] for i in face])
            face_data.append((centroid, face))

        face_data.sort(key=lambda x : x[0])
        faces = np.array([face for _, face in face_data])
        faces = torch.from_numpy(faces)
        return faces
    
    def _normalize_mesh(self, mesh):
        """
        Normalize vertices of mesh so that it fits inside a cube bounding box of size 1.0 and zero centers it.

        Parameters:
            mesh (trimesh.Trimesh): Input mesh
        """

        # Center the mesh at the origin
        center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-center)

        # Scale so the largest dimension becomes 1
        scale = mesh.extents.max()
        mesh.apply_scale(1.0 / scale)

        return mesh
    
    def _lex_sort_verts(self, face: torch.Tensor, all_vertices: torch.Tensor):
        """lexicographically sorts vertices present in individual faces
            Params:
                Face (np.array): 1D list of vertices forming a single face
                all_vertices (np.array): list of all vertices present in mesh rearranged as zyx
        """
        
        face_vertices = np.array([all_vertices[vert] for vert in face])
        
        sorted_idx = np.lexsort((face_vertices[:, 2], face_vertices[:,1], face_vertices[:, 0]))
        
        return face_vertices[sorted_idx]

    def quantize(self, sequence: torch.Tensor):
        "converts float values to discrete int bins"
        return (torch.clamp(torch.floor((sequence + (self.box_dim / 2)) * (self.bins / self.box_dim)), 0, self.bins - 1)).to(dtype=torch.int64)

    def dequantize(self, tokens: torch.Tensor):
        "converts integer bins to float values"
        return (tokens.float() / (self.bins - 1)) * self.box_dim - (self.box_dim / 2)
    
    def encode(self, mesh_path: str):

        mesh = trimesh.load(mesh_path)

        mesh = self._normalize_mesh(mesh)

        face_list = self._extract_faces_bot_top(mesh)

        sorted_faces_verts = torch.from_numpy(np.array([self._lex_sort_verts(face, mesh.vertices) for face in face_list]))
        
        #arrange vertices as x,y,z -> z,y,x. z represents vertical axis.
        sorted_faces_verts = sorted_faces_verts[:, :, [2,1,0]]

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
    
def _test_mesh_tokenizer():
    mesh_path = R"/root/Nvidia-Meshtron-Pytorch/mesh/suzanne.obj"
    tokenizer = MeshTokenizer(bins=128)
    tokens = tokenizer.encode(mesh_path)
    print(tokens.size())
    coord = tokenizer.decode(tokens)
    print(coord.size())
    print(tokens)
    print(coord)

if __name__ == "__main__":
    _test_mesh_tokenizer()
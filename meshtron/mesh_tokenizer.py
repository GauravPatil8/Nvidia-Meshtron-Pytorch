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

    
    def __extract_faces_bot_top(self, mesh: trimesh.Trimesh):
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
    
    def __normalize_verts_to_box(self, file_path: str):
        """
        Normalize vertices of mesh so that it fits inside a cube bounding box of size 1.0 and zero centers it.

        Parameters:
            mesh (trimesh.Trimesh): Input mesh
        """

        vertices = self.__get_vertices(file_path)
        min_coord = np.min(vertices, axis=0)
        max_coord = np.max(vertices, axis=0)

        # Center of bounding box
        center = (max_coord + min_coord) / 2.0
        vertices -= center  # shift to zero-center
    
        dimension = max_coord - min_coord
        
        scale = 1.0 / np.max(dimension)

        vertices *= scale

        return torch.from_numpy(vertices)
    
    def __lex_sort_verts(self, face: torch.Tensor, all_vertices: torch.Tensor):
        """lexicographically sorts vertices present in individual faces
            Params:
                Face (np.array): 1D list of vertices forming a single face
                all_vertices (np.array): list of all vertices present in mesh rearranged as zyx
        """
        
        face_vertices = np.array([all_vertices[vert] for vert in face])
        
        sorted_idx = np.lexsort((face_vertices[:, 2], face_vertices[:,1], face_vertices[:, 0]))
        
        return face_vertices[sorted_idx]
    
    def __get_vertices(self, obj_file: str):
        if not os.path.exists(obj_file):
            raise FileNotFoundError(f"File not found {obj_file}")
        vertices = []
        with open(obj_file, 'r') as obj:
            for line in obj:
                line = line.strip()

                if not line or line.startswith('#'):
                    continue

                parts  = line.split()

                if parts[0] == 'v':
                    vertices.append(parts[1:])
        return np.array(vertices, dtype=float)

    def quantize(self, sequence: torch.Tensor):
        "converts float values to discrete int bins"
        return (torch.clamp(torch.floor((sequence + (self.box_dim / 2)) * (self.bins / self.box_dim)), 0, self.bins - 1)).to(dtype=torch.int64)

    def dequantize(self, token):
        "converts integer bins to float values"
        return (token.float() / (self.bins - 1)) * self.box_dim - (self.box_dim / 2)
    
    def encode(self, mesh_path: str):

        mesh = trimesh.load(mesh_path)

        vertices = self.__normalize_verts_to_box(mesh_path)
       
        mesh.vertices = vertices

        face_list = self.__extract_faces_bot_top(mesh)

        #arrange vertices as x,y,z -> z,y,x. z represents vertical axis.
        vertices = vertices[:, [2,1,0]]

        sorted_faces_verts = torch.from_numpy(np.array([self.__lex_sort_verts(face, vertices) for face in face_list]))

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
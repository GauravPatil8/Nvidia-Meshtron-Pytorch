import os
import torch
import random
import trimesh
import numpy as np
from pathlib import Path

def load_obj(filepath):
    """Load vertices and faces from an OBJ file."""
    vertices = []
    faces = []
    header = []

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("v "):  # vertex
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)
            elif line.startswith("f "):  # face
                parts = line.strip().split()
                face = [int(p.split("/")[0]) for p in parts[1:]]
                faces.append(face)
            else:
                header.append(line.strip())  # keep comments, object names, etc.
    return np.array(vertices), faces, " "

def random_transform(vertices):
    """Apply random rotation + independent scaling to vertices."""
    # Random rotation angles
    angles = np.radians(np.random.uniform(-5, 5, size=3))
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])

    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])

    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]])

    # Combined rotation
    R = Rz @ Ry @ Rx

    # Apply scaling first, then rotation
    transformed = (vertices @ R) 

    return transformed

def save_obj(filepath, vertices, faces, header):
    """Save vertices and faces back into an OBJ file."""
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    with open(filepath, "w") as f:
        for line in header:
            f.write(line + "\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("s 0\n")
        for face in faces:
            f.write("f " + " ".join(str(idx) for idx in face) + "\n")

def normalize_verts_to_box(file_path: str):
    """
    Normalize vertices of mesh so that it fits inside a cube bounding box of size 1.0 and zero centers it.

    Parameters:
        mesh (trimesh.Trimesh): Input mesh
    """

    vertices = get_vertices(file_path)
    min_coord = np.min(vertices, axis=0)
    max_coord = np.max(vertices, axis=0)

    # Center of bounding box
    center = (max_coord + min_coord) / 2.0
    vertices -= center  # shift to zero-center
  
    dimension = max_coord - min_coord
    
    scale = 1.0 / np.max(dimension)

    vertices *= scale

    return torch.from_numpy(vertices)

# def map_to_bins(points: np.array, bins: int, box_dim: float = 1.0):
#     "converts float values to discrete int32 bins"
#     return np.clip(np.floor((points + (box_dim / 2)) * (bins / box_dim)), 0, bins - 1)


def get_mesh_stats(obj_file: str):
  "Returns len(faces) & quad ratio"
  if not os.path.exists(obj_file):
      raise FileNotFoundError(f"File not found {obj_file}")
  faces_count = 0
  quad_count = 0
  with open(obj_file, 'r') as obj:
      for line in obj:
          line = line.strip()

          if not line or line.startswith('#'):
              continue

          parts = line.split()
          if parts[0] == 'f':
              faces_count += 1
              if len(parts[1:]) == 4:
                  quad_count += 1
  return faces_count, (quad_count / faces_count)

def get_vertices(obj_file: str):
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

# def extract_faces_bot_top(mesh: trimesh.Trimesh):
#     "Returns list of faces arranged from bottom to top"

#     faces = mesh.faces
#     vertices = mesh.vertices

#     face_data = []
#     for face in faces:
#         centroid = np.mean([vertices[i][2] for i in face])
#         face_data.append((centroid, face))

#     face_data.sort(key=lambda x : x[0])
#     faces = np.array([face for _, face in face_data])
#     faces = torch.from_numpy(faces)
#     return faces

# def lex_sort_verts(face: torch.Tensor, all_vertices: torch.Tensor):
#     """lexicographically sorts vertices present in individual faces
#         Params:
#             Face (np.array): 1D list of vertices forming a single face
#             all_vertices (np.array): list of all vertices present in mesh rearranged as zyx
#     """
    
#     face_vertices = np.array([all_vertices[vert] for vert in face])
    
#     sorted_idx = np.lexsort((face_vertices[:, 2], face_vertices[:,1], face_vertices[:, 0]))
    
#     return face_vertices[sorted_idx]

def get_max_seq_len(data_dir: str):
    "Returns the max seq len the model will recieve"
    max_seq_len = float('-inf')
    for file in os.listdir(data_dir):
        mesh = trimesh.load(os.path.join(data_dir, file), file_type='obj')
        max_seq_len = max(max_seq_len, (len(mesh.faces) * 9)) # mesh.faces returns triangular faces
    return int(max_seq_len + 9) # adding 9 as for preserving hourglass structure we are adding 9 special tokens (<sos> or <eos>)

def add_gaussian_noise(x:torch.Tensor, mean:float, std: float):
    noise = torch.normal(mean=mean, std=std, size=x.shape, device=x.device)
    return x + noise

def set_zero_vector(points: torch.Tensor, rate: float, size: int):
    """Selects random points and sets them to zero vector
        Params:
            points: Tensor 
            rate: % of conversion
            size: size of zero array
    """
    n_points = len(points)
    k = int(n_points * rate)
    indices = torch.randperm(n_points)[:k]
    points[indices] = torch.zeros(size)
    return points
    
def write_obj(point_cloud, file_name):
    """
        point_cloud: vertex point cloud generated by model
        file_name: full path to store the file. eg: "artifacts/gen/cube.obj"
    """
    vertex_list_length = 0
    vertex_set = set()
    vertex_list = []
    face_list = []
    tri_list = []
    file_path = Path(file_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    for point in point_cloud:
        if len(tri_list) == 3:
            face_list.append(tri_list)
            tri_list = []
        
        if point in vertex_set:
            vertex_idx = vertex_list.index(point) + 1 # 1 based indexing
            tri_list.append(vertex_idx)
        else:
            vertex_list.append(point)
            vertex_list_length+=1
            tri_list.append(vertex_list_length)


    with open(file_path, 'w') as f:
        for v in vertex_list:
            f.write(f"v {' '.join(str(f'{coord:.6f}') for coord in v)}\n") 

        for face in face_list:
            f.write(f"f {' '.join(str(idx) for idx in face)}\n")       

# def get_point_cloud_data(mesh_path: str):
#     mesh = trimesh.load_mesh(mesh_path, file_type = 'obj')
#     vertices = normalize_verts_to_box(mesh_path)

#     mesh.vertices = vertices

#     #sampling points on the surface of the bounded mesh (N, 3)
#     point_cloud, face_indices = trimesh.sample.sample_surface(mesh, 8192//2)

#     #point cloud & point normals
#     point_cloud = torch.from_numpy(point_cloud).to(dtype=torch.float32)
#     point_normals = torch.from_numpy(mesh.face_normals[face_indices]).to(dtype=torch.float32)

#     # augmentation
#     point_cloud = add_gaussian_noise(point_cloud, mean=0.0, std=0.01) #according to paper: mean = 0.0, std = 0.01
#     point_normals = add_gaussian_noise(point_normals, mean=0.0, std=0.03)
#     point_normals = set_zero_vector(points=point_normals, rate=0.3, size=point_normals.shape[1])

#     points = torch.cat((point_cloud, point_normals), dim=1)

#     return points, point_cloud
import os
import torch
import trimesh
import numpy as np

def load_obj(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(mesh_path, file_type = 'obj')
    return mesh

def save_obj(mesh: trimesh.Trimesh, dir_path: str, index: int):
    os.makedirs(dir_path, exist_ok=True)
    mesh.export(os.path.join(dir_path, f"{index}.obj"))

def apply_random_rotation(mesh: trimesh.Trimesh, max_angle_degree: int = 180) -> None:
    "Applies Random Euler rotation to the mesh"

    euler_x = np.random.uniform(-max_angle_degree, max_angle_degree)
    euler_y = np.random.uniform(-max_angle_degree, max_angle_degree)
    euler_z = np.random.uniform(-max_angle_degree, max_angle_degree)

    mesh.apply_transform(trimesh.transformations.rotation_matrix(euler_x, [1,0,0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(euler_y, [0,1,0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(euler_z, [0,0,1]))

def apply_random_scale(mesh: trimesh.Trimesh, scale_ranges: dict = None) -> None:
    """
        Applies random scaling on each axis

        Parameters:
            mesh: trimesh.Trimesh object
            scale_ranges: dict with 'x', 'y', 'z' keys containing (min, max) tuples
    
    """

    if scale_ranges is None:
        scale_ranges = {
            'x': (0.5, 2.0),
            'y': (0.3, 1.5),
            'z': (0.2, 3.0)
        }
    
    scale_x = np.random.uniform(*scale_ranges['x'])
    scale_y = np.random.uniform(*scale_ranges['y'])
    scale_z = np.random.uniform(*scale_ranges['z'])

    mesh.apply_scale([scale_x, scale_y, scale_z])

def apply_random_transformations(mesh: trimesh.Trimesh, max_angle_degree: int = 180, scale_ranges: dict = None) -> None:
    "Applies random rotation and scaling for each axis"

    apply_random_rotation(mesh, max_angle_degree)
    apply_random_scale(mesh, scale_ranges)

def normalize_mesh_to_bbox(mesh: trimesh.Trimesh, box_size_dim: float = 1.0):
    """
    Normalize a mesh so that it fits inside a cube bounding box of size `box_size_dim`.

    Parameters:
        mesh (trimesh.Trimesh): Input mesh
        box_size_dim (float): Target cube side length
    """
    vertices = mesh.vertices

    # current bounding box
    min_coord = np.min(vertices, axis=0)
    max_coord = np.max(vertices, axis=0)
    current_size = max_coord - min_coord

    # avoid divide by zero
    current_size[current_size == 0] = 1e-9  

    # uniform scaling factor
    scale_factor = (box_size_dim / np.max(current_size))

    # scale around the center of bbox
    center = (min_coord + max_coord) / 2.0
    normalized_vertices = (vertices - center) * scale_factor

    mesh.vertices = normalized_vertices
    return mesh

def map_to_bins(points: np.array, bins: int, box_dim: float = 1.0):
    "converts float values to discrete int32 bins"
    return np.clip(np.floor((points + (box_dim / 2)) * (bins / box_dim)), 0, bins - 1)


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
    return np.array(vertices)

def extract_faces_bot_top(mesh: trimesh.Trimesh):
    "Returns list of faces arranged from bottom to top"

    faces = mesh.faces
    vertices = mesh.vertices

    face_data = []
    for face in faces:
        centroid = np.mean([vertices[i][2] for i in face])
        face_data.append((centroid, face))

    face_data.sort(key=lambda x : x[0])

    return torch.tensor([face for _, face in face_data])

def lex_sort_verts(face: torch.Tensor, all_vertices: torch.Tensor):
    """lexicographically sorts vertices present in individual faces
        Params:
            Face (np.array): 1D list of vertices forming a single face
            all_vertices (np.array): list of all vertices present in mesh rearranged as zyx
    """
    
    face_vertices = np.array([all_vertices[vert] for vert in face])
    
    sorted_idx = np.lexsort((face_vertices[:, 2], face_vertices[:,1], face_vertices[:, 0]))
    
    return face_vertices[sorted_idx]



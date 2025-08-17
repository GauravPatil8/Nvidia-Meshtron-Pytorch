import os
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

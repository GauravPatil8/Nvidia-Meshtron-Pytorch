import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import numpy as np
from src.utils.common import get_path
from torch.utils.data import Dataset, DataLoader
from src.utils.data import load_obj, normalize_mesh_to_bbox


class MeshDataset(Dataset):
    def __init__(self, dataset_dir: str, num_points: int = 2048):
        """
            Returns normalized and scaled to bounding box mesh
            Parameters:
                dataset_dir = location of stored data.
                num_points = number of points to sample on the mesh
        """
        self.data_dir = dataset_dir
        self.num_points = num_points
        self.files = [get_path(root, file) for root, dirs, files in os.walk(dataset_dir) for file in files]
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):

        mesh = load_obj(self.files[index])

        # normalize and centralize the mesh
        bounded_mesh = normalize_mesh_to_bbox(mesh, 1.0)

        #sampling points on the surface of the bounded mesh
        point_cloud = bounded_mesh.sample(self.num_points)

        #Rearrange xyz to yzx as mentioned in the paper
        point_cloud = point_cloud[:, [1,2,0]]

        #lexsort
        sorted_idx = np.lexsort(point_cloud[:, 2], point_cloud[:, 1], point_cloud[:,0])
        point_cloud = point_cloud[sorted_idx]

        # return tuple(tensor of points (x, y, z), path of the original model)
        return (torch.from_numpy(point_cloud), self.files[index])
    

if __name__ == '__main__':
    
    dataset = MeshDataset(R"C:\Padhai\implementation_research_papers\meshtron\artifacts\dataset", num_points=2048)

    # Test __len__
    print("Dataset size:", len(dataset))

    # Test one item
    points, path = dataset[32]
    print("First mesh points shape:", points.shape)
    print("File path:", path)

    # Check range of points
    print("Min:", points.min(axis=0))
    print("Max:", points.max(axis=0))
    print("Mean:", points.mean(axis=0))

    # Wrap in dataloader
    # loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # for batch in loader:
    #     x, paths = batch
    #     print("Batch points:", x.shape)
    #     print("Paths:", paths)
    #     break
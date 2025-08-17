import os
from utils.common import get_path
from utils.data import apply_random_transformations, save_obj, load_obj

class Ingestion:
    def __init__(self, len_dataset: int, mesh_dir: str, dataset_dir: str):
        """ 
            Initializes Ingestion stage

            Parameters:
                len_dataset (int): The total length of dataset.(must be divisible by number of meshes)
                mesh_dir (str): Path of the folder where primitive meshes are stored.
                dataset_dir (str): Path of the folder where the transformed meshes will be stored for training.
        """
        self.len_dataset = len_dataset
        self.mesh_dir = mesh_dir
        self.dataset = dataset_dir

    def run(self):
        #list of paths of all meshes
        self.meshes = [get_path(self.mesh_dir, path) for path in os.listdir(self.mesh_dir)]

        #Creating same number of instances for each mesh
        assert  self.len_dataset % len(self.meshes) == 0 , "length of dataset should be divisible by count of meshes"

        instances_per_mesh = self.len_dataset // len(self.meshes)

        for mesh_path in self.meshes:
            dir_path = get_path(self.dataset, os.path.splitext(os.path.basename(mesh_path))[0])
            mesh = load_obj(mesh_path).copy()

            for index in range(instances_per_mesh):
                apply_random_transformations(mesh)
                save_obj(mesh, dir_path, index+1) #obj staring from index 1 [cube\1.obj, cube\2.obj]
                

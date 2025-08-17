import os
from utils.common import get_path
from config import ConfigurationManager
from utils.data import rotateandscale, save_obj

class Ingestion:
    def __init__(self):
        self.config = ConfigurationManager.ingestion_config()

    def run(self):
        #list of paths of all meshes
        self.meshes = [get_path(self.config.meshes, path) for path in os.listdir(self.config.meshes)]

        #Creating same number of instances for each mesh
        assert  self.config.len_dataset % len(self.meshes) == 0 , "length of dataset should be divisible by count of meshes"

        instances_per_mesh = self.config.len_dataset // len(self.meshes)

        for mesh in self.meshes:
            mesh_path = mesh
            dir_path = get_path(self.config.dataset, os.path.splitext(os.path.basenamename(mesh_path))[0])

            for index in range(instances_per_mesh):
                mesh = rotateandscale(mesh_path)
                save_obj(mesh, dir_path, index+1) #obj staring from 1 [1.obj, 2.obj]
                

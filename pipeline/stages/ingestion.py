import os
import logging
from tqdm import tqdm
from pipeline.utils.common import get_path, logger_init
from pipeline.utils.data import save_obj, load_obj, random_transform
from pipeline.config_entities import IngestionConfig

class Ingestion:
    def __init__(self, config: IngestionConfig):
        """ 
            Initializes Ingestion stage

            Parameters:
                len_dataset (int): The total length of dataset.(must be divisible by number of meshes)
                mesh_dir (str): Path of the folder where primitive meshes are stored.
                dataset_dir (str): Path of the folder where the transformed meshes will be stored for training.
        """
        self.len_dataset = config.dataset_len
        self.mesh_dir = config.meshes
        self.dataset = config.dataset_storage_dir
        self.exists = True if os.path.exists(config.dataset_storage_dir) else False

    def __str__(self):
        return f"Ingestion Stage f{Ingestion}"
    
    def run(self):
        logger = logger_init()

        #list of paths of all meshes
        self.meshes = [get_path(self.mesh_dir, path) for path in os.listdir(self.mesh_dir)]

        #Creating same number of instances for each mesh
        assert  self.len_dataset % len(self.meshes) == 0 , "length of dataset should be divisible by count of meshes"

        instances_per_mesh = self.len_dataset // len(self.meshes)

        if not self.exists:
            logger.info(f"Creating dataset at : {self.dataset}")
            try:
                for mesh_path in tqdm(self.meshes, desc='Meshes'):
                    dir_name = os.path.splitext(os.path.basename(mesh_path))[0]
                    dir_path = get_path(self.dataset, dir_name)
                    vertices, faces, header = load_obj(mesh_path)

                    for index in tqdm(range(instances_per_mesh), desc=f"Instances for {dir_name}", leave=False):
                        # transformed_vertices = random_transform(vertices)

                        save_obj(get_path(dir_path, f"{index+1}.obj"), vertices, faces, header) #obj staring from index 1 [cube\1.obj, cube\2.obj]
            except Exception as e:
                logger.info(f"An unexpected error occured during ingestion: {e}")

        else:
            logger.info(f"Dataset already exists, skipping ingestion stage")
                

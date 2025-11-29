import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from pipeline.utils.data import get_mesh_stats, get_point_cloud_data
from pipeline.config import ConfigurationManager
# from pipeline.utils.model import get_model
import requests
import json
import os
import trimesh
class MeshGenerationViewer:
    def __init__(self, point_cloud, rotation_speed=2.0):
        """
        Initialize the 3D mesh generation viewer.
        
        Args:
            point_cloud: Nx3 numpy array of point cloud coordinates
            rotation_speed: Rotation speed in degrees per update (default: 2.0)
        """
        self.point_cloud = point_cloud
        self.vertices = []
        self.faces = []
        self.generation_started = False
        self.rotation_speed = rotation_speed
        self.current_angle = 0
        
        # Setup the figure
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('white')
        self.fig.patch.set_facecolor('white')
        
        # Initial display of point cloud
        self.point_scatter = self.ax.scatter(
            point_cloud[:, 0], 
            point_cloud[:, 1], 
            point_cloud[:, 2],
            c='blue', 
            marker='o', 
            s=1,
            alpha=0.5
        )
        
        # Set axis properties
        self._setup_axes()
        
        # Set isometric view (elevation=35.264, azimuth=45)
        self.ax.view_init(elev=35.264, azim=45)
        
        # Initialize mesh collection
        self.mesh_collection = None
        
        plt.ion()  # Interactive mode
        plt.show()
        plt.pause(2)  # Display point cloud for 2 seconds
        
        # Start continuous rotation animation
        self.anim = FuncAnimation(
            self.fig, 
            self._rotate, 
            interval=50,  # Update every 50ms
            cache_frame_data=False
        )
    
    def _setup_axes(self):
        """Setup axis limits and labels"""
        # Calculate bounds from point cloud
        mins = self.point_cloud.min(axis=0)
        maxs = self.point_cloud.max(axis=0)
        margin = (maxs - mins) * 0.1
        
        self.ax.set_xlim(mins[0] - margin[0], maxs[0] + margin[0])
        self.ax.set_ylim(mins[1] - margin[1], maxs[1] + margin[1])
        self.ax.set_zlim(mins[2] - margin[2], maxs[2] + margin[2])
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Mesh Generation Progress')
    
    def _rotate(self, frame):
        """Animation function for continuous rotation"""
        self.current_angle = (self.current_angle + self.rotation_speed) % 360
        self.ax.view_init(elev=35.264, azim=45 + self.current_angle)
        return []
    
    def add_tokens(self, tokens):
        """
        Add tokens to build the mesh progressively.
        Tokens are vertex coordinates: [x, y, z, x, y, z, ...]
        Every 9 tokens (3 vertices) forms one face.
        
        Args:
            tokens: List or array of coordinate values
        """
        # Remove point cloud when generation starts
        if not self.generation_started:
            self.point_scatter.remove()
            self.generation_started = True
        
        # Add tokens to vertices buffer
        for token in tokens:
            self.vertices.append(token)
            
            # Check if we have a complete face (9 coordinates = 3 vertices)
            if len(self.vertices) >= 9:
                # Extract the last 3 vertices
                v1 = self.vertices[-9:-6]
                v2 = self.vertices[-6:-3]
                v3 = self.vertices[-3:]
                
                # Add face (indices reference the vertex list)
                face_idx = len(self.faces)
                self.faces.append([v1, v2, v3])
                
                # Update visualization
                self._update_mesh()
                
                # Clear processed vertices to save memory (optional)
                # Keep all vertices if you need them later
    
    def add_face(self, v1, v2, v3):
        """
        Directly add a complete face with three vertices.
        
        Args:
            v1, v2, v3: Lists or arrays of [x, y, z] coordinates
        """
        # Remove point cloud when generation starts
        if not self.generation_started:
            self.point_scatter.remove()
            self.generation_started = True
        
        self.faces.append([v1, v2, v3])
        self._update_mesh()
    
    def _update_mesh(self):
        """Update the mesh visualization"""
        # Remove old mesh collection
        if self.mesh_collection is not None:
            self.mesh_collection.remove()
        
        # Create new mesh collection
        self.mesh_collection = Poly3DCollection(
            self.faces,
            facecolors='lightblue',
            edgecolors='black',
            alpha=0.8,
            linewidths=0.5
        )
        
        self.ax.add_collection3d(self.mesh_collection)
        
        # Update the display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def finalize(self):
        """Finalize the visualization (disable interactive mode)"""
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    mesh_dir = ConfigurationManager.dataset_config().original_mesh_dir
    monkey_obj = os.path.join(mesh_dir, 'suzanne.obj')
    cube_obj = os.path.join(mesh_dir,'cube.obj')
    cone_obj = os.path.join(mesh_dir,'cone.obj')
    sphere_obj = os.path.join(mesh_dir,'sphere.obj')
    torus_obj = os.path.join(mesh_dir,'torus.obj')
    
    selected_mesh = sphere_obj

    points , point_cloud = get_point_cloud_data(selected_mesh)
    points = points.unsqueeze(0)

    face_count, quad_ratio = get_mesh_stats(selected_mesh)
    face_count = torch.tensor([face_count], dtype=torch.float32)
    quad_ratio = torch.tensor([quad_ratio], dtype=torch.float32) 
    url = "http://localhost:8000/stream"
    
    # Create viewer
    viewer = MeshGenerationViewer(point_cloud.cpu().numpy(), rotation_speed=2.0)

    payload = {
        "point_cloud": points.tolist(),
        "face_count": face_count.item(),
        "quad_ratio": quad_ratio.item()
    }
    face_list = []
    vert_list = []
    coord_buffer = []

    with requests.post(url, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if not line:
                continue
            
            coord = float(line.decode())
            coord_buffer.append(coord)

            # If we got one full vertex (3 floats)
            if len(coord_buffer) == 3:
                vert = coord_buffer[:]  # copy
                coord_buffer = []

                vert[0], vert[2] = vert[2], vert[0]

                vert_list.append(vert)

            # If we got 3 vertices => one face
            if len(vert_list) == 3:
                v1, v2, v3 = vert_list

                viewer.add_face(v1, v2, v3)
                vert_list = []
    
    print("Mesh generation complete!")
    viewer.finalize()
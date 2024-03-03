import trimesh
import numpy as np

from utils.transform import Transform


class Object:
    def __init__(self, filename: str, scale: float = 1.0):
        self.filename = filename
        self.mesh = trimesh.load(self.filename)
        self.mesh.apply_scale(scale)
        self.mesh_center = self.mesh.centroid
        self.world_frame_pose = Transform(to_frame='object')
        self.scale = scale

    def elevate_mesh_to_zero(self):
        min_z = self.mesh.vertices[:, 2].min()
        if min_z < 0:
            elevation_translation = Transform(translation=np.array([0, 0, -min_z]), to_frame='object')
            self.mesh = elevation_translation.apply_to_mesh(self.mesh)

    def set_transform(self, t: Transform):
        # Reload and reapply scaling to the mesh
        self.mesh = trimesh.load(self.filename)
        self.mesh.apply_scale(self.scale)
        self.world_frame_pose = t
        self.apply_transform()

    def apply_transform(self):
        self.mesh = self.world_frame_pose.apply_to_mesh(self.mesh)
        self.mesh_center = self.mesh.centroid

    def scale_mesh(self, scale_factor: float):
        self.mesh.apply_scale(scale_factor)
        self.scale *= scale_factor
        self.mesh_center = self.mesh.centroid

    def rotate_x(self, angle: float):
        rotation_matrix = Transform.rotation_matrix_x(angle)
        self.world_frame_pose = self.world_frame_pose.compose(rotation_matrix)
        self.apply_transform()

    def rotate_y(self, angle: float):
        rotation_matrix = Transform.rotation_matrix_y(angle)
        self.world_frame_pose = self.world_frame_pose.compose(rotation_matrix)
        self.apply_transform()

    def rotate_z(self, angle: float):
        rotation_matrix = Transform.rotation_matrix_z(angle)
        self.world_frame_pose = self.world_frame_pose.compose(rotation_matrix)
        self.apply_transform()

    def get_center(self):
        return self.mesh_center

    def get_mesh(self):
        return self.mesh
import trimesh
import numpy as np

from transform import Transform


class Object:
    def __init__(self, filename: str):
        self.mesh = trimesh.load(filename)
        self.elevate_mesh_to_zero()
        self.mesh_center = self.mesh.centroid
        self.world_frame_pose = Transform(to_frame='object')

    def elevate_mesh_to_zero(self):
        min_z = self.mesh.vertices[:, 2].min()
        if min_z < 0:
            elevation_translation = Transform(translation=np.array([0, 0, -min_z]), to_frame='object')
            self.mesh = elevation_translation.apply_to_mesh(self.mesh)

    def set_pose(self, rotation, translation):
        self.world_frame_pose = Transform(rotation, translation, to_frame='object')
        self.apply_transform()

    def apply_transform(self):
        self.mesh = self.world_frame_pose.apply_to_mesh(self.mesh)
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

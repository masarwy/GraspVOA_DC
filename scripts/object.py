import trimesh

from transform import Transform


class Object:
    def __init__(self, filename):
        self.mesh = trimesh.load(filename)
        self.mesh_center = self.mesh.centroid
        self.world_frame_pose = Transform(to_frame='object')

    def set_pose(self, rotation, translation):
        self.world_frame_pose = Transform(rotation, translation, to_frame='object')
        self.apply_transform()

    def apply_transform(self):
        self.mesh = self.world_frame_pose.apply_to_mesh(self.mesh)
        self.mesh_center = self.mesh.centroid

    def get_center(self):
        return self.mesh_center

    def get_mesh(self):
        return self.mesh

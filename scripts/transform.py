import numpy as np


class Transform:
    def __init__(self, rotation=np.eye(3), translation=np.zeros(3), from_frame='world', to_frame='object'):
        self.rotation = rotation
        self.translation = translation
        self.from_frame = from_frame
        self.to_frame = to_frame
        if not self.is_valid():
            raise ValueError("Invalid transformation parameters.")

    def is_valid(self):
        if not np.allclose(np.dot(self.rotation, self.rotation.T), np.eye(3), atol=1e-6):
            return False

        if not np.isclose(np.linalg.det(self.rotation), 1.0, atol=1e-6):
            return False

        if self.translation.shape != (3,) or not np.issubdtype(self.translation.dtype, np.number):
            return False

        return True

    def inverse(self):
        inv_rotation = self.rotation.T
        inv_translation = -inv_rotation @ self.translation
        return Transform(inv_translation, inv_rotation, self.to_frame, self.from_frame)

    @staticmethod
    def rotation_matrix_x(angle):
        c, s = np.cos(angle), np.sin(angle)
        return Transform(rotation=np.array([[1, 0, 0], [0, c, -s], [0, s, c]]))

    @staticmethod
    def rotation_matrix_y(angle):
        c, s = np.cos(angle), np.sin(angle)
        return Transform(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))

    @staticmethod
    def rotation_matrix_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        return Transform(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))

    def get_transformation_matrix(self):
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T

    def combine(self, other_transform):
        new_rotation = self.rotation @ other_transform.rotation
        new_translation = self.rotation @ other_transform.translation + self.translation
        return Transform(new_translation, new_rotation, self.from_frame, other_transform.to_frame)

    def apply_to_mesh(self, mesh):
        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(self.get_transformation_matrix())
        return transformed_mesh

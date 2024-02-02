import numpy as np

from transform import Transform


class Camera:
    def __init__(self, radius, target=np.array([0, 0, 0]), up_vector=np.array([0, 0, 1])):
        self.radius = radius
        self.target = target
        self.up_vector = up_vector

    def spherical_to_cartesian(self, theta, phi):
        x = self.radius * np.sin(theta) * np.cos(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(theta)
        return np.array([x, y, z])

    def look_at(self, camera_position):
        z_axis = camera_position - self.target
        z_axis /= np.linalg.norm(z_axis)

        # Check if the up vector and the z_axis are parallel
        if np.allclose(z_axis, self.up_vector) or np.allclose(z_axis, -self.up_vector):
            up_vector_adjusted = np.array([self.up_vector[1], -self.up_vector[0], self.up_vector[2]])
        else:
            up_vector_adjusted = self.up_vector

        x_axis = np.cross(up_vector_adjusted, z_axis)
        if np.linalg.norm(x_axis) == 0:
            x_axis = np.array([1, 0, 0])
        else:
            x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
        return rotation_matrix

    def generate_poses(self, n_theta, n_phi):
        for theta in np.linspace(0, np.pi / 2, n_theta):
            for phi in np.linspace(0, 2 * np.pi, n_phi):
                camera_position = self.spherical_to_cartesian(theta, phi)
                rotation_matrix = self.look_at(camera_position)
                pose = Transform(rotation_matrix, camera_position, 'object', 'camera')
                yield pose

    def generate_camera_poses(self, n_samples_theta, n_samples_phi):
        """
        Generate camera poses around the northern hemisphere of a sphere using strategic sampling.
        This is a generator function that yields camera pose matrices one by one.

        Args:
        - radius (float): The radius of the sphere.
        - n_samples_theta (int): Number of samples in the theta dimension (polar angle), only in the northern hemisphere.
        - n_samples_phi (int): Number of samples in the phi dimension (azimuthal angle).
        """
        # Linear distribution for theta
        theta_values = np.linspace(0, np.pi / 2, n_samples_theta)

        for theta in theta_values:
            # Adjust the number of phi samples based on theta to have denser sampling away from the pole
            if theta == 0:
                # At the pole, sample only one point
                phi_samples = [0]
            else:
                # As theta increases, increase the number of phi samples
                phi_density = int(np.ceil(n_samples_phi * np.sin(theta)))
                phi_samples = np.linspace(0, 2 * np.pi, phi_density, endpoint=False)

            for phi in phi_samples:
                camera_position = self.spherical_to_cartesian(theta, phi)
                rotation_matrix = self.look_at(camera_position)
                pose = Transform(rotation_matrix, camera_position, 'object', 'camera')
                yield pose

    def create_camera_pose_from_x(self):
        theta, phi = np.pi / 2, 0
        camera_position = self.spherical_to_cartesian(theta, phi)
        rotation_matrix = self.look_at(camera_position)
        pose = Transform(rotation_matrix, camera_position, 'object', 'camera')
        return pose

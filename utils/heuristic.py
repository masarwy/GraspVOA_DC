import numpy as np

from utils.transform import Transform, Point3D
import matplotlib.pyplot as plt


class CameraVisibilityHeuristic:
    def __init__(self, poi: Point3D, optimal_distance: float, camera_params: dict):
        """
        Initialize the heuristic with parameters specific to the task.

        :param poi: The Point of Interest in the workspace (numpy array).
        :param optimal_distance: The optimal distance from the camera to the PoI.
        :param camera_params: Parameters of the camera, such as FoV, resolution, etc.
        """
        self.poi = poi
        self.optimal_distance = optimal_distance
        if camera_params is None:
            self.camera_params = {'fov_horizontal_rad': 65, 'fov_vertical_rad': 40, 'image_width': 1280,
                                  'image_height': 720}

        else:
            self.camera_params = camera_params

    def __call__(self, camera_transform: Transform) -> float:
        """
        Evaluate the heuristic for a given camera pose.

        :param camera_pose: The pose of the camera in the workspace frame (numpy array or suitable representation).
        :return: A score representing the quality of the camera pose based on the heuristic.
        """

        distance_score = self._calculate_distance_score(camera_transform)
        visibility_score = self._calculate_visibility_score(camera_transform)

        # Combine scores, consider weighted sum
        total_score = distance_score + visibility_score

        return total_score

    def _calculate_distance_score(self, camera_pose: Transform) -> float:
        """
        Calculate the distance score based on the camera's distance to the PoI.

        :param camera_pose: The pose of the camera.
        :return: The distance score.
        """
        distance = np.linalg.norm(self.poi - np.array(camera_pose.translation))
        score = max(0, 1 - abs(distance - self.optimal_distance) / self.optimal_distance)
        return score

    def poi_in_camera_frame(self, camera_pose_transform: Transform):
        """
        Transforms a point of interest from the world frame to the camera frame.

        :param camera_pose_transform: A Transform object representing the pose of the camera in the world frame.
        :return: The coordinates of the PoI in the camera frame.
        """
        world_to_camera_transform = camera_pose_transform.inverse()

        poi_world_homogeneous = np.append(self.poi, 1)

        poi_camera_homogeneous = world_to_camera_transform.get_transformation_matrix() @ poi_world_homogeneous

        poi_camera = poi_camera_homogeneous[:3] / poi_camera_homogeneous[3]

        return poi_camera

    def plot_poi_on_image(self, camera_pose_transform: Transform):
        # For debugging purpose
        poi_camera_frame = self.poi_in_camera_frame(camera_pose_transform)
        fov_horizontal_rad = np.radians(self.camera_params['fov_horizontal_rad'])
        fov_vertical_rad = np.radians(self.camera_params['fov_vertical_rad'])
        image_width = self.camera_params['image_width']
        image_height = self.camera_params['image_height']

        # Calculate the position of the PoI on the image plane
        x_on_image = (poi_camera_frame[0] / poi_camera_frame[2]) * (image_width / 2) / np.tan(
            fov_horizontal_rad / 2) + (image_width / 2)
        y_on_image = (poi_camera_frame[1] / poi_camera_frame[2]) * (image_height / 2) / np.tan(fov_vertical_rad / 2) + (
                image_height / 2)

        # Create an empty image
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        # Plot the PoI on the image
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.scatter([x_on_image], [image_height - y_on_image], color='red',
                    s=50)  # Inverting y-axis to match image coordinates
        plt.title('PoI in Camera Space')
        plt.axis('off')
        plt.show()

    def _calculate_visibility_score(self, camera_pose_transform: Transform) -> float:
        """
        Calculate the visibility score based on how centered the PoI is within the camera's FoV.

        :param camera_pose: The pose of the camera.
        :return: The visibility score.
        """
        poi_camera_frame = self.poi_in_camera_frame(camera_pose_transform)

        fov_horizontal_rad = np.radians(self.camera_params['fov_horizontal_rad'])
        fov_vertical_rad = np.radians(self.camera_params['fov_vertical_rad'])
        image_width = self.camera_params['image_width']
        image_height = self.camera_params['image_height']

        x_on_image = (poi_camera_frame[0] / poi_camera_frame[2]) * (image_width / 2) / np.tan(
            fov_horizontal_rad / 2) + (image_width / 2)
        y_on_image = (poi_camera_frame[1] / poi_camera_frame[2]) * (image_height / 2) / np.tan(fov_vertical_rad / 2) + (
                image_height / 2)

        x_center = image_width / 2
        y_center = image_height / 2

        deviation_x = abs(x_on_image - x_center)
        deviation_y = abs(y_on_image - y_center)

        max_deviation_x = image_width / 2
        max_deviation_y = image_height / 2
        visibility_score = 1 - (deviation_x / max_deviation_x + deviation_y / max_deviation_y) / 2
        visibility_score = max(0, visibility_score)

        return visibility_score

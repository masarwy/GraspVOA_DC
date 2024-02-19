import numpy as np
import yaml
from scipy.stats import vonmises, multivariate_normal

from transform import Pose


def calculate_weights(particles: np.ndarray, observation: any) -> None:
    for i, particle in enumerate(particles):
        position = particle[2:4].astype(np.float64)

        distance = np.linalg.norm(position - observation)

        particles[i, -1] = 1 / (distance + 1e-9)

    total_weight = np.sum(particles[:, -1].astype(np.float64))
    particles[:, -1] = particles[:, -1].astype(np.float64) / total_weight


class BeliefSpaceModel:
    def __init__(self, standard_poses_file: str):
        with open(standard_poses_file, 'r') as file:
            data = yaml.safe_load(file)
            self.standard_poses = data['standard_poses']

        self.pose_categories = {category: pose['probability'] for category, pose in self.standard_poses.items()}

        self.angle_kappas = {category: 1 for category in self.pose_categories.keys()}
        self.angle_distributions = {category: vonmises(kappa=self.angle_kappas[category]) for category in
                                    self.pose_categories.keys()}

        self.position_distributions = {
            category: multivariate_normal(
                mean=pose['position_offset'][:2],
                cov=[[0.02, 0], [0, 0.02]])
            for category, pose in self.standard_poses.items()
        }

    def display_parameters(self, title: str = "Model Parameters") -> None:
        print(title)
        for category in self.pose_categories.keys():
            print(f"\nCategory: {category}")
            print(f"Pose Probability: {self.pose_categories[category]:.2f}")
            print(f"Angle Distribution Kappa: {self.angle_kappas[category]:.2f}")
            print(f"Position Distribution Mean: {self.position_distributions[category].mean}")
            print(f"Position Distribution Covariance: \n{self.position_distributions[category].cov}")

    def sample_particles(self, n_particles: int = 1000) -> np.ndarray:
        particles = []
        for category, prob in self.pose_categories.items():
            n_samples = int(n_particles * prob)
            for _ in range(n_samples):
                angle_sample = self.angle_distributions[category].rvs()
                position_sample = self.position_distributions[category].rvs()
                particles.append((category, angle_sample, *position_sample, 0.))
        return np.array(particles)

    def update_model(self, particles: np.ndarray) -> None:
        """
        Updates the belief space
        :param particles: (category, angle, x, y, weight)
        :return:
        """
        self.display_parameters("Parameters Before Update")

        for category in self.pose_categories.keys():
            category_particles = particles[particles[:, 0] == category]

            if len(category_particles) > 0:
                angles = category_particles[:, 1].astype(np.float64)
                weights = category_particles[:, 4].astype(np.float64)

                weighted_angle_mean = np.average(angles, weights=weights)

                angle_diffs_squared = (angles - weighted_angle_mean) ** 2
                weighted_variance = np.sum(weights * angle_diffs_squared) / np.sum(weights)

                self.angle_distributions[category] = vonmises(kappa=1 / weighted_variance)
                self.angle_kappas[category] = 1 / weighted_variance

                positions = category_particles[:, 2:4].astype(np.float64)

                weighted_positions = np.average(positions, axis=0, weights=weights)

                position_diffs = positions - weighted_positions
                weighted_covariance = np.dot((position_diffs.T * weights), position_diffs) / np.sum(weights)

                self.position_distributions[category] = multivariate_normal(mean=weighted_positions,
                                                                            cov=weighted_covariance)

        self.display_parameters("Parameters After Update")

    def particle_to_6d_pose(self, category: str, angle: float, x: float, y: float) -> Pose:
        standard_pose = self.standard_poses[category]

        orientation = np.array(standard_pose['orientation'])
        orientation[2] += angle

        position_offset = np.array(standard_pose['position_offset'])
        position = np.array([x, y, 0]) + position_offset

        pose = Pose(0, 0, 0, 0, 0, 0)
        pose.x = position[0]
        pose.y = position[1]
        pose.z = position[2]
        pose.Rx = orientation[0]
        pose.Ry = orientation[1]
        pose.Rz = orientation[2]

        return pose


# Example usage:
# model = BeliefSpaceModel(standard_poses_file='../data/objects/ENDSTOP/standard_poses.yaml')
# particles = model.sample_particles()
# observation = np.array([0.5, 0.5])
# calculate_weights(particles, observation)
# model.update_model(particles)

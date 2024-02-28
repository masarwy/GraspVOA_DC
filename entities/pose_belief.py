import numpy as np
import yaml
from scipy.stats import vonmises, multivariate_normal

from utils.transform import Pose, Point3D


def calculate_weights(particles: np.ndarray, observation: any) -> None:
    for i, particle in enumerate(particles):
        position = particle[2:4].astype(np.float64)

        distance = np.linalg.norm(position - observation)

        particles[i, -1] = 1 / (distance + 1e-9)

    total_weight = np.sum(particles[:, -1].astype(np.float64))
    particles[:, -1] = particles[:, -1].astype(np.float64) / total_weight


def normalize_angle(angle):
    """
    Normalize an angle in radians to be within the range -π to π.

    Parameters:
    angle (float): The angle in radians to normalize.

    Returns:
    float: The normalized angle within the range -π to π.
    """
    # Normalize the angle to be within the range 0 to 2π
    angle = angle % (2 * np.pi)

    # Adjust angles greater than π to be within the range -π to 0
    if angle > np.pi:
        angle -= 2 * np.pi

    return angle


class BeliefSpaceModel:
    def __init__(self, standard_poses_file: str, poi: 'Point3D'):
        """
        To update to sequential scenario, this class should be updated such that it saves loc in von mises distribution
        as I did for the kappas.
        :param standard_poses_file: a yaml file where the standard category poses are saved
        :param poi: point of interest
        """
        self.poi = poi

        with open(standard_poses_file, 'r') as file:
            data = yaml.safe_load(file)
            self.standard_poses = data['standard_poses']

        self.pose_categories = {category: pose['probability'] for category, pose in self.standard_poses.items()}

        self.pose_categories_prior = self.pose_categories.copy()

        # Initialize model parameters and their priors
        self.angle_kappas = {category: 0.001 for category in self.pose_categories.keys()}
        self.angle_kappas_prior = self.angle_kappas.copy()

        self.position_means = {category: np.array([poi.x, poi.y]) for category in self.pose_categories.keys()}
        self.position_means_prior = self.position_means.copy()

        self.position_covariances = {
            category: np.array([[0.15, 0], [0, 0.15]]) for category in self.pose_categories.keys()
        }
        self.position_covariances_prior = self.position_covariances.copy()

        self.angle_distributions = {category: vonmises(kappa=self.angle_kappas[category]) for category in
                                    self.pose_categories.keys()}
        self.position_distributions = {
            category: multivariate_normal(mean=self.position_means[category], cov=self.position_covariances[category])
            for category in self.pose_categories.keys()
        }

    def reset(self):
        self.pose_categories = {category: pose['probability'] for category, pose in self.standard_poses.items()}
        self.pose_categories_prior = self.pose_categories.copy()

        self.angle_kappas = {category: 0.001 for category in self.pose_categories.keys()}
        self.angle_kappas_prior = self.angle_kappas.copy()  # Reset the priors as well
        self.angle_distributions = {category: vonmises(kappa=self.angle_kappas[category]) for category in
                                    self.pose_categories.keys()}

        self.position_means = {category: np.array([self.poi.x, self.poi.y]) for category in self.pose_categories.keys()}
        self.position_means_prior = self.position_means.copy()  # Reset the priors as well

        initial_covariance = np.array([[0.15, 0], [0, 0.15]])
        self.position_covariances = {category: initial_covariance for category in self.pose_categories.keys()}
        self.position_covariances_prior = self.position_covariances.copy()  # Reset the priors as well

        self.position_distributions = {
            category: multivariate_normal(mean=self.position_means[category], cov=self.position_covariances[category])
            for category in self.pose_categories.keys()
        }

    def update_model(self, particles: np.ndarray, prior_strength: float = 0.25) -> None:
        # self.display_parameters("Parameters Before Update")

        epsilon = 1e-3
        total_weight = np.sum(particles[:, 4].astype(np.float64))  # Sum of all particle weights
        updated_category_weights = {category: 0.0 for category in self.pose_categories.keys()}

        for category in self.pose_categories.keys():
            category_particles = particles[particles[:, 0] == category]
            if len(category_particles) > 0:
                angles = category_particles[:, 1].astype(np.float64)
                weights = category_particles[:, 4].astype(np.float64)

                updated_category_weights[category] = np.sum(weights)

                # Calculate weighted angle mean considering circular statistics
                weighted_angle_mean = np.arctan2(np.sum(weights * np.sin(angles)), np.sum(weights * np.cos(angles)))

                # Update angle parameters considering prior
                angle_diffs = np.arctan2(np.sin(angles - weighted_angle_mean), np.cos(angles - weighted_angle_mean))
                angle_diffs_squared = angle_diffs ** 2
                weighted_variance = np.sum(weights * angle_diffs_squared) / np.sum(weights)
                new_kappa = 0.001 if weighted_variance == 0 else 1 / weighted_variance
                self.angle_kappas[category] = (prior_strength * self.angle_kappas_prior[category]) + (
                        (1 - prior_strength) * new_kappa)

                # Update position parameters considering prior
                positions = category_particles[:, 2:4].astype(np.float64)
                weighted_positions = np.average(positions, axis=0, weights=weights)
                position_diffs = positions - weighted_positions
                weighted_covariance = np.dot((position_diffs.T * weights), position_diffs) / np.sum(weights)
                regularized_covariance = weighted_covariance + np.eye(weighted_covariance.shape[0]) * epsilon
                self.position_means[category] = (prior_strength * self.position_means_prior[category]) + (
                        (1 - prior_strength) * weighted_positions)
                self.position_covariances[category] = (prior_strength * self.position_covariances_prior[category]) + (
                        (1 - prior_strength) * regularized_covariance)

                # Update distributions
                self.angle_distributions[category] = vonmises(kappa=self.angle_kappas[category],
                                                              loc=weighted_angle_mean)
                self.position_distributions[category] = multivariate_normal(mean=self.position_means[category],
                                                                            cov=self.position_covariances[category])

        # Update category probabilities and priors if needed
        if total_weight > 0:
            for category in self.pose_categories.keys():
                updated_probability = updated_category_weights[category] / total_weight
                self.pose_categories[category] = (prior_strength * self.pose_categories_prior[category]) + (
                        (1 - prior_strength) * updated_probability)
        else:
            for category in self.pose_categories.keys():
                self.pose_categories[category] = 1.0 / len(self.pose_categories)

        self.angle_kappas_prior = self.angle_kappas.copy()
        self.position_means_prior = self.position_means.copy()
        self.position_covariances_prior = self.position_covariances.copy()
        self.pose_categories_prior = self.pose_categories.copy()

        # self.display_parameters("Parameters After Update")

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

    def particle_to_6d_pose(self, category: str, angle: float, x: float, y: float) -> Pose:
        standard_pose = self.standard_poses[category]

        orientation = np.array(standard_pose['orientation'])
        orientation[2] += angle
        orientation[2] = normalize_angle(orientation[2])

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

    def calculate_likelihood(self, category: str, angle: float, x: float, y: float) -> float:
        if category not in self.pose_categories:
            raise ValueError(f"Unknown category: {category}")

        category_prob = self.pose_categories[category]

        angle_likelihood = self.angle_distributions[category].pdf(angle)

        position_likelihood = self.position_distributions[category].pdf([x, y])

        total_likelihood = category_prob * angle_likelihood * position_likelihood

        return total_likelihood

    def calculate_log_likelihood(self, category: str, angle: float, x: float, y: float) -> float:
        if category not in self.pose_categories:
            raise ValueError(f"Unknown category: {category}")

        log_category_prob = np.log(self.pose_categories[category])

        raw_log_angle_likelihood = self.angle_distributions[category].logpdf(angle)
        min_log_likelihood = -3
        max_log_likelihood = 0
        log_angle_likelihood = np.clip(raw_log_angle_likelihood, min_log_likelihood, max_log_likelihood)

        log_position_likelihood = self.position_distributions[category].logpdf([x, y])

        total_log_likelihood = log_category_prob + log_angle_likelihood + log_position_likelihood

        return total_log_likelihood

# Example usage:
# model = BeliefSpaceModel(standard_poses_file='../data/objects/ENDSTOP/standard_poses.yaml')
# particles = model.sample_particles()
# observation = np.array([0.5, 0.5])
# calculate_weights(particles, observation)
# model.update_model(particles)

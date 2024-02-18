import numpy as np
from scipy.stats import vonmises, multivariate_normal


def calculate_weights(particles, observation):
    weights = np.ones(len(particles))
    for i, particle in enumerate(particles):
        # Example: simple distance for weighting, replace with your actual observation model
        distance = np.linalg.norm(particle[2:].astype(np.float64) - observation)
        weights[i] = 1 / (distance + 1e-9)  # Add a small constant to avoid division by zero
    return weights / np.sum(weights)


class BeliefSpaceModel:
    def __init__(self):
        self.pose_categories = {'upright': 0.33, 'upside_down': 0.33, 'lying_on_edge': 0.34}
        self.angle_kappas = {category: 1 for category in self.pose_categories.keys()}
        self.angle_distributions = {category: vonmises(kappa=1) for category in self.pose_categories.keys()}
        self.position_distributions = {category: multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]]) for category in
                                       self.pose_categories.keys()}

    def display_parameters(self, title="Model Parameters"):
        print(title)
        for category in self.pose_categories.keys():
            print(f"\nCategory: {category}")
            print(f"Pose Probability: {self.pose_categories[category]:.2f}")
            print(f"Angle Distribution Kappa: {self.angle_kappas[category]:.2f}")
            print(f"Position Distribution Mean: {self.position_distributions[category].mean}")
            print(f"Position Distribution Covariance: \n{self.position_distributions[category].cov}")

    def sample_particles(self, n_particles=1000):
        particles = []
        for category, prob in self.pose_categories.items():
            n_samples = int(n_particles * prob)
            for _ in range(n_samples):
                angle_sample = self.angle_distributions[category].rvs()
                position_sample = self.position_distributions[category].rvs()
                particles.append((category, angle_sample, *position_sample))
        return np.array(particles)

    def update_model(self, particles, weights):
        # Display parameters before the update
        self.display_parameters("Parameters Before Update")

        # Update angle and position distributions based on weighted particles
        for category in self.pose_categories.keys():
            category_particles = particles[particles[:, 0] == category]
            category_weights = weights[particles[:, 0] == category]

            if len(category_particles) > 0:
                # Compute weighted mean for angles
                angles = category_particles[:, 1].astype(np.float64)
                weighted_angle_mean = np.average(angles, weights=category_weights)

                # Compute weighted variance for angles manually
                angle_diffs_squared = (angles - weighted_angle_mean) ** 2
                weighted_variance = np.sum(category_weights * angle_diffs_squared) / np.sum(category_weights)

                # Update angle distribution parameters for this category
                self.angle_distributions[category] = vonmises(kappa=1 / weighted_variance)
                self.angle_kappas[category] = 1 / weighted_variance

                # Extract x and y positions and form a 2D positions array
                x_positions = category_particles[:, 2].astype(np.float64)
                y_positions = category_particles[:, 3].astype(np.float64)
                positions = np.stack((x_positions, y_positions), axis=-1)  # Shape will be (n_samples, 2)

                # Compute weighted positions
                weighted_positions = np.average(positions, axis=0, weights=category_weights)

                # Compute weighted covariance for positions
                position_diffs = positions - weighted_positions
                weighted_covariance = np.dot((position_diffs.T * category_weights), position_diffs) / np.sum(
                    category_weights)

                # Update position distribution parameters for this category
                self.position_distributions[category] = multivariate_normal(mean=weighted_positions,
                                                                            cov=weighted_covariance)

        # Display parameters after the update
        self.display_parameters("Parameters After Update")


# Example usage:
model = BeliefSpaceModel()
particles = model.sample_particles()
observation = np.array([0.5, 0.5])
weights = calculate_weights(particles, observation)
model.update_model(particles, weights)

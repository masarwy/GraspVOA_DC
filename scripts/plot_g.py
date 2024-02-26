import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def plot_gaussians(initial_mean, initial_cov, updated_mean, updated_cov, grid_size=3):
    # Create a grid of points
    x, y = np.mgrid[initial_mean[0] - grid_size:initial_mean[0] + grid_size:.01,
           initial_mean[1] - grid_size:initial_mean[1] + grid_size:.01]
    pos = np.dstack((x, y))

    # Create multivariate normal distributions
    rv_initial = multivariate_normal(initial_mean, initial_cov)
    rv_updated = multivariate_normal(updated_mean, updated_cov)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # Plot the initial Gaussian
    axs[0].contourf(x, y, rv_initial.pdf(pos), levels=40, cmap='viridis')
    axs[0].set_title('Initial Gaussian Distribution')
    axs[0].grid(True)

    # Plot the updated Gaussian
    axs[1].contourf(x, y, rv_updated.pdf(pos), levels=40, cmap='viridis')
    axs[1].set_title('Updated Gaussian Distribution')
    axs[1].grid(True)

    plt.show()


# Initial mean and covariance
initial_mean = [-1.2, -1.]
initial_cov = [[0.02, 0.], [0., 0.02]]

# Updated mean and covariance
updated_mean = [-1.2, -1.]
updated_cov = [[0.001, 0.0], [0.0, 0.001]]

# Call the function with the initial and updated parameters
plot_gaussians(initial_mean, initial_cov, updated_mean, updated_cov)

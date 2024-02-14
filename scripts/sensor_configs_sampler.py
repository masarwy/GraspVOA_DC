import numpy as np

from heuristic import CameraVisibilityHeuristic
from transform import Transform, Point3D
from ur5e_fk import forward


def heuristic_driven_sampling(camera_in_ee: Transform, camera_params: dict, joint_limits: np.ndarray, poi: Point3D,
                              dist_res: int = 1000, n_samples: int = 10):
    sampled_configs = []
    scores = []
    heuristic = CameraVisibilityHeuristic(poi=poi, optimal_distance=70., camera_params=camera_params)

    for _ in range(dist_res):
        config = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1])
        ee_in_world = forward(config)
        camera_pose = ee_in_world.compose(camera_in_ee).to_pose()
        score = heuristic(camera_pose)
        sampled_configs.append(camera_pose)
        scores.append(score)

    probabilities = np.exp(scores) / np.sum(np.exp(scores))

    selected_indices = np.random.choice(range(dist_res), size=n_samples, p=probabilities).tolist()
    selected_configs = [sampled_configs[i] for i in selected_indices]

    return selected_configs

import numpy as np

from heuristic import CameraVisibilityHeuristic
from transform import Transform, Point3D
from ur5e_fk import forward


def heuristic_driven_sampling(camera_in_ee: Transform, camera_params: dict, joint_limits: np.ndarray, poi: Point3D,
                              dist_res: int = 1000, n_samples: int = 10):
    sampled_configs = []
    sampled_configs_joints = []
    scores = []
    heuristic = CameraVisibilityHeuristic(poi=poi, optimal_distance=70., camera_params=camera_params)

    for _ in range(dist_res):
        config = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1])
        ee_in_world = forward(config)
        if ee_in_world.translation[2] < 0.1 or ee_in_world.translation[0] >= -0.1 or ee_in_world.translation[1] >= -0.1:
            continue
        if np.linalg.norm(ee_in_world.translation) > 10.:
            continue
        camera_in_world = ee_in_world.compose(camera_in_ee)
        score = heuristic(camera_in_world)
        sampled_configs.append(camera_in_world.to_pose_axis_angle())
        scores.append(score)
        sampled_configs_joints.append(config)

    probabilities = np.exp(scores) / np.sum(np.exp(scores))

    selected_indices = probabilities.argsort()[-n_samples:].tolist()
    selected_configs = [sampled_configs[i] for i in selected_indices]
    selected_configs_joints = [sampled_configs_joints[i] for i in selected_indices]
    selected_scores = [scores[i] for i in selected_indices]

    # selected_indices = np.random.choice(range(dist_res), size=n_samples, p=probabilities).tolist()
    # selected_configs = [sampled_configs[i] for i in selected_indices]

    return selected_configs, selected_configs_joints, selected_scores

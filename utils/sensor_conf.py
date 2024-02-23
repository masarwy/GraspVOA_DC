import numpy as np
import yaml

from utils.transform import Transform, Point3D, Pose
from sensor_configs_sampler import heuristic_driven_sampling


def dict_to_pose(data):
    return Pose(x=data['x'], y=data['y'], z=data['z'], Rx=data['Rx'], Ry=data['Ry'], Rz=data['Rz'])


def read_sensor_configs(sensor_configs_file: str):
    with open(sensor_configs_file, 'r') as f:
        loaded_data = yaml.safe_load(f)

    sensor_p_q = {}
    for sensor_id, sensor_data in loaded_data.items():
        pose = dict_to_pose(sensor_data['pose'])
        joint = sensor_data['joints']
        sensor_p_q[sensor_id] = {'pose': pose, 'joints': joint}
    return sensor_p_q


def pose_to_dict(pose: Pose) -> dict:
    return {
        'x': float(pose.x), 'y': float(pose.y), 'z': float(pose.z),
        'Rx': float(pose.Rx), 'Ry': float(pose.Ry), 'Rz': float(pose.Rz)
    }


if __name__ == '__main__':
    camera_params_file = '../data/camera_params.yaml'
    with open(camera_params_file, 'r') as file:
        config = yaml.safe_load(file)
    camera_params = config['camera_params']
    yfov = np.radians(camera_params['fov_vertical_rad'])
    aspect_ratio = camera_params['image_width'] / camera_params['image_height']

    camera_in_ee = Transform(rotation=np.eye(3), translation=np.array([0, -0.105, 0.0395]), from_frame='EE',
                             to_frame='camera')
    joint_limits = np.zeros((6, 2))
    joint_limits[:, 1] = np.array([2 * np.pi] * 6)

    poi = Point3D(-1.2, -1., 0)

    poses, joints, scores = heuristic_driven_sampling(camera_in_ee=camera_in_ee, camera_params=camera_params,
                                                      joint_limits=joint_limits, poi=poi)

    data_to_save = {}

    for i, (pose, joint, score) in enumerate(zip(poses, joints, scores), start=1):
        pose_dict = pose_to_dict(pose)
        joint_list = [float(j) for j in joint]
        sensor_id = f"sensor_c_{i}"
        data_to_save[sensor_id] = {'pose': pose_dict, 'joints': joint_list, 'score': float(score)}

    with open('../data/poses_and_joints.yaml', 'w') as file:
        yaml.safe_dump(data_to_save, file, default_flow_style=False)

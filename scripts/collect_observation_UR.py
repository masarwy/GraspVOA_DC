import urx
from PIL import Image
import numpy as np

from utils.sensor_conf import read_sensor_configs
from entities.real_camera import RealCamera

if __name__ == '__main__':

    robot_ip = '192.168.2.2'
    sensor_poses_file = '../data/poses_and_joints.yaml'
    sensor_p_q = read_sensor_configs(sensor_poses_file)

    object_id = 'FLASK'

    rob = urx.Robot(robot_ip, use_rt=True)
    real_camera = RealCamera('../data/camera_params.yaml')
    object_pose_id = 3
    for sensor_id in sensor_p_q.keys():
        sen_id = int(sensor_id[-1])
        j_state = sensor_p_q[sensor_id]['joints']
        rob.movej(j_state, vel=5., acc=2.)
        depth, color_image = real_camera.get_images()
        np.save('../data/objects/' + object_id + f'/img/lab/di_{sen_id}_{object_pose_id}.npy', depth)
        np.save('../data/objects/' + object_id + f'/img/lab/c_{sen_id}_{object_pose_id}.npy', color_image)

        image_min = depth.min()
        image_max = depth.max()

        # Normalize to 0-1 range
        normalized_image = (depth - image_min) / (image_max - image_min)

        # Scale to 0-255 and convert to uint8
        scaled_image = (normalized_image * 255).astype(np.uint8)
        di_image = Image.fromarray(scaled_image)
        di_image.save('../data/objects/' + object_id + f'/img/lab/di_{sen_id}_{object_pose_id}.png')
        c_image = Image.fromarray(color_image)
        c_image.save('../data/objects/' + object_id + f'/img/lab/c_{sen_id}_{object_pose_id}.png')

    # Disconnect from the robot
    rob.close()

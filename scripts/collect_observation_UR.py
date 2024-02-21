import urx
import matplotlib.pyplot as plt

from sensor_conf import read_sensor_configs
from real_camera import RealCamera


if __name__ == '__main__':

    robot_ip = '192.168.2.2'
    sensor_poses_file = '../data/poses_and_joints.yaml'
    sensor_p_q = read_sensor_configs(sensor_poses_file)

    rob = urx.Robot(robot_ip, use_rt=True)
    real_camera = RealCamera('../data/camera_params.yaml')

    for sensor_id in sensor_p_q.keys():
        j_state = sensor_p_q[sensor_id]['joints']
        rob.movej(j_state, vel=5., acc=2.)
        depth_image, color_image = real_camera.get_images()
        plt.imshow(color_image)
        plt.show()
        plt.imshow(depth_image, cmap='gray')
        plt.show()

    # Disconnect from the robot
    rob.close()

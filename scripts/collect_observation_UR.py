import urx
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt

from sensor_conf import read_sensor_configs
from transform import Transform


def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    pipeline.start(config)

    return pipeline, config


def get_images(pipeline):
    align_to = rs.stream.color
    align = rs.align(align_to)

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return depth_image, color_image


if __name__ == '__main__':

    robot_ip = '192.168.2.2'
    sensor_poses_file = '../data/poses_and_joints.yaml'
    sensor_p_q = read_sensor_configs(sensor_poses_file)

    rob = urx.Robot(robot_ip, use_rt=True)
    pipeline, config = init_realsense()

    for sensor_id in sensor_p_q.keys():
        j_state = sensor_p_q[sensor_id]['joints']
        print(Transform.from_rv(sensor_p_q[sensor_id]['pose']))
        rob.movej(j_state, vel=5., acc=2.)
        depth_image, color_image = get_images(pipeline)
        plt.imshow(color_image)
        plt.show()
        plt.imshow(depth_image, cmap='gray')
        plt.show()

    # Disconnect from the robot
    rob.close()

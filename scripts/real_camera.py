import pyrealsense2 as rs
import yaml
import numpy as np


class RealCamera:
    def __init__(self, camera_params_file: str):
        with open(camera_params_file, 'r') as file:
            data = yaml.safe_load(file)
        width = data['camera_params']['image_width']
        height = data['camera_params']['image_height']

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)

        self.pipeline.start(self.config)

    def get_images(self):
        align_to = rs.stream.color
        align = rs.align(align_to)

        frames = self.pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image

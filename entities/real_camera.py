import pyrealsense2 as rs
import yaml
import numpy as np
import cv2


def preprocess_depth_image(depth_img: np.ndarray, filter_size: int = 5, threshold: float = 2):
    smooth_img = cv2.GaussianBlur(depth_img, (filter_size, filter_size), 0)

    non_zero_elements = smooth_img[smooth_img != 0]
    non_zero_sorted = np.sort(non_zero_elements)
    median_val = np.median(non_zero_sorted)

    lower_bound = median_val / threshold
    upper_bound = median_val * threshold

    non_zero_mask = smooth_img != 0

    depth_img_clipped = np.zeros_like(smooth_img)
    depth_img_clipped[non_zero_mask] = np.clip(smooth_img[non_zero_mask], lower_bound, upper_bound)
    depth_img_clipped[depth_img_clipped <= lower_bound] = 0

    return depth_img_clipped


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
        # Align the color frame to the depth frame
        align_to = rs.stream.depth
        align = rs.align(align_to)

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            raise ValueError("Could not obtain aligned frames from the RealSense device")

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image

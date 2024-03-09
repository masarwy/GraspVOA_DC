import matplotlib.pyplot as plt
import pyrealsense2 as rs
import yaml
import numpy as np

with open('../data/camera_params.yaml', 'r') as file:
    data = yaml.safe_load(file)
width = data['camera_params']['image_width']
height = data['camera_params']['image_height']

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)

pipeline.start(config)

frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()
depth = np.asanyarray(depth_frame.get_data())
color = np.asanyarray(color_frame.get_data())

align_to = rs.stream.depth
align = rs.align(align_to)

aligned_frames = align.process(frames)
depth_frame_al = aligned_frames.get_depth_frame()
color_frame_al = aligned_frames.get_color_frame()
depth_al = np.asanyarray(depth_frame_al.get_data())
color_al = np.asanyarray(color_frame_al.get_data())


id = 6

# save depth, color as npy array
np.save(f'../tmp_scratch/depth{id}.npy', depth)
np.save(f'../tmp_scratch/depth{id}_aligned.npy', depth_al)
np.save(f'../tmp_scratch/color{id}.npy', color)
np.save(f'../tmp_scratch/color{id}_aligned.npy', color_al)

plt.imshow(color_al)
plt.show()
plt.imshow(color)
plt.show()

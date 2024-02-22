import pyrealsense2 as rs
import math

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
config = rs.config()

# Start the pipeline
pipeline.start(config)

# Get the device that is connected to the pipeline
device = pipeline.get_active_profile().get_device()

# Get the depth sensor from the connected device
depth_sensor = device.first_depth_sensor()

# Get the depth scale (depth scale is the conversion from depth pixels to meters)
depth_scale = depth_sensor.get_depth_scale()

# Get the depth stream profile and load it to a variable
depth_profile = rs.video_stream_profile(pipeline.get_active_profile().get_stream(rs.stream.depth))

# Get intrinsic parameters of the depth stream
intrinsics = depth_profile.get_intrinsics()

# Print intrinsic parameters
print("Intrinsic parameters of the depth stream:")
print("Width:", intrinsics.width)
print("Height:", intrinsics.height)
print("PPX:", intrinsics.ppx)
print("PPY:", intrinsics.ppy)
print("FX:", intrinsics.fx)
print("FY:", intrinsics.fy)
print("Distortion Model:", intrinsics.model)
print("Distortion Coefficients:", intrinsics.coeffs)

# Intrinsic parameters
width = intrinsics.width  # obtained from intrinsics of the camera
height = intrinsics.height  # obtained from intrinsics of the camera
fx = intrinsics.fx  # focal length along x-axis
fy = intrinsics.fy  # focal length along y-axis

# Calculate horizontal and vertical FoV in radians
fov_horizontal_rad = 2 * math.atan(width / (2 * fx))
fov_vertical_rad = 2 * math.atan(height / (2 * fy))

# Print the results
print(f"Horizontal FoV in radians: {fov_horizontal_rad}")
print(f"Vertical FoV in radians: {fov_vertical_rad}")

fov_horizontal_deg = math.degrees(fov_horizontal_rad)
fov_vertical_deg = math.degrees(fov_vertical_rad)

# Print the results in degrees
print(f"Horizontal FoV in degrees: {fov_horizontal_deg}")
print(f"Vertical FoV in degrees: {fov_vertical_deg}")

# Stop the pipeline
pipeline.stop()

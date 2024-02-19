import math

standard_poses = {
    "C1": {
        "probability": 0.1,
        "orientation": [177, 0, 0],  # Degrees
        "position_offset": [0, 0, 51.15]  # Millimeters
    },
    "C2": {
        "probability": 0.05,
        "orientation": [-73, 0, 0],  # Degrees
        "position_offset": [0, 0, 28.51]  # Millimeters
    },
    "C3": {
        "probability": 0.35,
        "orientation": [0, 90, 0],  # Degrees
        "position_offset": [0, 0, 15.7]  # Millimeters
    },
    "C4": {
        "probability": 0.4,
        "orientation": [0, -90, 0],  # Degrees
        "position_offset": [0, 0, 15.7]  # Millimeters
    },
    "C5": {
        "probability": 0.1,
        "orientation": [90, 0, 0],  # Degrees
        "position_offset": [0, 0, 32.49]  # Millimeters
    }
}

# Convert each pose
for pose in standard_poses.values():
    # Convert orientation from degrees to radians
    pose['orientation'] = [math.radians(angle) for angle in pose['orientation']]
    # Convert position offset from millimeters to meters
    pose['position_offset'] = [offset / 1000 for offset in pose['position_offset']]

# Display the converted poses
for category, pose in standard_poses.items():
    print(f"{category}: Orientation (radians) = {pose['orientation']}, Position Offset (meters) = {pose['position_offset']}")
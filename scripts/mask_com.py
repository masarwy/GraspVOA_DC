import numpy as np
import matplotlib.pyplot as plt

sensor_id = 5
pose_id = 4

mask1 = plt.imread(f'../data/objects/ENDSTOP/img/lab/mask_{sensor_id}_{pose_id}.png')
mask2 = np.load(f'../data/objects/ENDSTOP/img/gen/di_{sensor_id}_{pose_id}.npy')
mask1 = mask1 != 0.
mask2 = mask2 != 0

combined_image = np.zeros((*mask1.shape, 3), dtype=np.uint8)

# Set mask1 to red (255, 0, 0) and mask2 to blue (0, 0, 255)
combined_image[mask1] = [255, 0, 0]  # Red
combined_image[mask2] = [0, 0, 255]  # Blue

overlap = mask1 & mask2  # Find overlapping areas
combined_image[overlap] = [255, 0, 255]  # Purple

plt.imshow(combined_image)
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()

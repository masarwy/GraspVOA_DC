import numpy as np
import matplotlib.pyplot as plt

object_id = 'MOUSE'
sensor_id = 5
pose_id = 2
pose_id_ = 0

mask1 = np.load(f'../data/objects/{object_id}/img/lab/mdi_{sensor_id}_{pose_id}.npy')
mask2 = np.load(f'../data/objects/{object_id}/img/gen/di_{sensor_id}_{pose_id}.npy')
mask1 = mask1 != 0.
mask2 = mask2 != 0.

combined_image = np.zeros((*mask1.shape, 3), dtype=np.uint8)

# Set mask1 to red (255, 0, 0) and mask2 to blue (0, 0, 255)
combined_image[mask1] = [255, 0, 0]  # Red
combined_image[mask2] = [0, 0, 255]  # Blue

overlap = mask1 & mask2  # Find overlapping areas
combined_image[overlap] = [255, 0, 255]  # Purple

plt.imshow(combined_image)
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()

c3 = 0.03 ** 2
cov_xy = np.cov(mask1.flatten(), mask2.flatten())[0, 1]
sigma_x = np.std(mask1)
sigma_y = np.std(mask2)
structural_similarity = (cov_xy + c3) / (sigma_x * sigma_y + c3)
# print(structural_similarity)

intersection = np.logical_and(mask1, mask2).sum()
union = np.logical_or(mask1, mask2).sum()
iou = intersection / union if union != 0 else 0
print(iou)

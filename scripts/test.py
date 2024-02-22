import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


sen_id = 0
obj_pose_id = 5

mask1 = np.load(f'../data/objects/ENDSTOP/img/lab/mdi_{sen_id}_{obj_pose_id}.npy').astype(bool)
mask2 = np.load(f'../data/objects/ENDSTOP/img/gen/di_{sen_id}_{obj_pose_id}.npy').astype(bool)
combined_image = np.zeros((*mask1.shape, 3), dtype=np.uint8)

# Set mask1 to red (255, 0, 0) and mask2 to blue (0, 0, 255)
combined_image[mask1] = [255, 0, 0]  # Red
combined_image[mask2] = [0, 0, 255]  # Blue

overlap = mask1 & mask2  # Find overlapping areas
combined_image[overlap] = [255, 0, 255]  # Purple

plt.imshow(combined_image)
plt.axis('off')
plt.show()

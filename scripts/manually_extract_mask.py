from skimage.segmentation import slic
from skimage.measure import regionprops
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2lab
import cv2

if __name__ == '__main__':
    sen_id = 0
    obj_pose_id = 5

    # Load the image
    image_lab = rgb2lab(io.imread(f'../data/objects/ENDSTOP/img/lab/c_{sen_id}_{obj_pose_id}.png'))

    # Apply SLIC segmentation
    segments = slic(image_lab, n_segments=280, compactness=10, enforce_connectivity=True)

    # Calculate properties for each superpixel
    props = regionprops(segments, intensity_image=image_lab)


    # Assuming we're looking for a red object
    # Define a function to check if a superpixel's color is within the range for red
    def is_red(region, red_threshold=30):
        # In the LAB color space, the 'a' channel represents color on a green-red axis
        # You may need to adjust the threshold based on your specific image
        return region.mean_intensity[1] > red_threshold


    # Find labels of superpixels that are red
    red_labels = [region.label for region in props if is_red(region)]

    # Create a mask for the object
    object_mask = np.isin(segments, red_labels)

    binary_image = (object_mask * 255).astype('uint8')

    # Save binary image
    cv2.imwrite(f'../data/objects/ENDSTOP/img/lab/mask_{sen_id}_{obj_pose_id}.png', binary_image)

    # Display the mask
    plt.imshow(object_mask, cmap='gray')
    plt.show()

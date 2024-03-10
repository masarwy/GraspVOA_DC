from skimage.segmentation import slic
from skimage.measure import regionprops
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2lab
from scipy.ndimage import label, find_objects, binary_fill_holes
import cv2

if __name__ == '__main__':
    object_id = 'MOUSE'
    sen_id = 5
    obj_pose_id = 2

    # Load the image
    image_lab = rgb2lab(io.imread(f'../data/objects/{object_id}/img/lab/c_{sen_id}_{obj_pose_id}.png'))

    # Apply SLIC segmentation
    segments = slic(image_lab, n_segments=400, compactness=10, enforce_connectivity=True)

    # Calculate properties for each superpixel
    props = regionprops(segments, intensity_image=image_lab)

    # Define a function to check if a superpixel's color is within the range for red
    def is_red(region, red_threshold=25):
        return region.mean_intensity[1] > red_threshold

    # Find labels of superpixels that are red
    red_labels = [region.label for region in props if is_red(region)]

    # Create a mask for the object
    object_mask = np.isin(segments, red_labels)

    # Label the regions and measure their areas
    labeled_array, num_features = label(object_mask)
    sizes = np.bincount(labeled_array.ravel())
    largest_label = sizes[1:].argmax() + 1  # The 0 label is for the background

    # Keep only the largest region
    largest_region = (labeled_array == largest_label)

    # Optionally, fill holes in the largest region
    largest_region_filled = binary_fill_holes(largest_region)

    # Smooth the edges of the largest region
    smoothed_region = cv2.GaussianBlur(largest_region_filled.astype('uint8'), (5, 5), 0)

    # Convert the largest region back to a binary image (0 and 255)
    binary_image = (smoothed_region * 255).astype('uint8')

    # Save binary image
    cv2.imwrite(f'../data/objects/{object_id}/img/lab/mask_{sen_id}_{obj_pose_id}.png', binary_image)

    # Display the mask
    plt.imshow(smoothed_region, cmap='gray')
    plt.show()

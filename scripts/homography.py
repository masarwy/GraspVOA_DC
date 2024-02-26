import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
import cv2


def normalise_depth(npy_file):
    image = np.load(npy_file)

    # Find the non-zero minimum and maximum values
    min_val = image[image > 0].min()
    max_val = image.max()

    # Apply normalization only to non-zero values
    return np.where(image > 0, (image - min_val) / (max_val - min_val), 0)


# Load the images
image1 = io.imread('../data/objects/ENDSTOP/img/lab/mask_0_0.png', as_gray=True)
image2 = io.imread('../data/objects/ENDSTOP/img/gen/di_0_2.png', as_gray=True)

# Apply threshold to create binary masks
thresh1 = threshold_otsu(image1)
thresh2 = threshold_otsu(image2)
binary1 = image1 > thresh1
binary2 = image2 > thresh2

# Find contours
contours1, _ = cv2.findContours(binary1.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(binary2.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Assume the largest contour corresponds to the object
# and calculate its centroid
def get_contour_centroid(contour):
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)


# Get the centroids of the largest contour in each image
centroid1 = get_contour_centroid(contours1[0])
centroid2 = get_contour_centroid(contours2[0])


# Using centroids to define the points for homography calculation
# We can take the centroid as one point, and at least three more points are needed
# We can take points at a fixed distance from the centroid in the four cardinal directions
def get_four_points(centroid):
    cx, cy = centroid
    return [(cx, cy), (cx, cy - 10), (cx + 10, cy), (cx, cy + 10), (cx - 10, cy)]


# Get points for both images
points1 = get_four_points(centroid1)
points2 = get_four_points(centroid2)

# Convert points to the proper format for homography calculation
src_pts = np.array(points1).astype('float32').reshape(-1, 1, 2)
dst_pts = np.array(points2).astype('float32').reshape(-1, 1, 2)

# Calculate the Homography
H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Identity matrix
I = np.identity(3)

# Calculate the penalty as the Frobenius norm of the difference
penalty = np.linalg.norm(H - I, 'fro')

print(H, penalty)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def process_real_depth_image(depth_image):
    depth_image = np.where(depth_image == 65535, np.nan, depth_image)

    min_valid_depth = 2/100
    max_valid_depth = 100/100

    depth_image_clipped = np.clip(depth_image, min_valid_depth, max_valid_depth)

    return np.nan_to_num(depth_image_clipped, nan=0.0)


if __name__ == "__main__":
    empty_scene = np.load('../data/empty_scene/lab/di_4.npy')
    object_scene = np.load('../data/objects/ENDSTOP/img/lab/di_4_0.npy')
    depth_difference = np.abs(object_scene - empty_scene)
    threshold = 0.01
    object_mask = depth_difference > threshold

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.title('Depth with Object')
    plt.imshow(object_scene, cmap='gray')
    plt.subplot(1, 4, 2)
    plt.title('Depth without Object')
    plt.imshow(empty_scene, cmap='gray')
    plt.subplot(1, 4, 3)
    plt.title('Depth Difference')
    plt.imshow(depth_difference, cmap='gray')
    plt.subplot(1, 4, 4)
    plt.title('Object Mask')
    plt.imshow(object_mask, cmap='gray')
    plt.show()

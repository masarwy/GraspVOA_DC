import glob
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
import numpy as np


def load_and_preprocess_image(file_path: str) -> np.ndarray:
    image = imread(file_path)
    return image


if __name__ == '__main__':
    file_pattern = '../data/img/di_*.png'
    image_files = sorted(glob.glob(file_pattern))

    if not image_files:
        raise ValueError("No image files found. Check the file pattern or directory.")

    images = [load_and_preprocess_image(file) for file in image_files]

    n_images = len(images)
    similarity_matrix = np.zeros((n_images, n_images))

    for i in range(n_images):
        for j in range(i, n_images):
            if i == j:
                similarity_matrix[i, j] = 1
            else:
                score, _ = ssim(images[i], images[j], full=True)
                similarity_matrix[i, j] = similarity_matrix[j, i] = score
    print(similarity_matrix)

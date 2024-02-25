import glob

from utils.similarity_strategy import *

if __name__ == '__main__':
    file_pattern = '../data/objects/ENDSTOP/img/gen/di_0_*.npy'
    image_files = sorted(glob.glob(file_pattern))

    if not image_files:
        raise ValueError("No image files found. Check the file pattern or directory.")

    n_images = len(image_files)
    similarity_matrix = np.zeros((n_images, n_images))

    sim_context = SimilarityContext(StructureTermStrategy())

    for i in range(n_images):
        for j in range(i, n_images):
            score = sim_context.compare_images(image_files[i], image_files[j], a_is_real=False)
            similarity_matrix[i, j] = similarity_matrix[j, i] = score

    n_images = similarity_matrix.shape[0]  # Assuming square matrix

    # Apply Softmax for each row
    softmax_matrix = np.zeros_like(similarity_matrix)

    for i in range(n_images):
        exp_row = np.exp(similarity_matrix[i] - np.max(similarity_matrix[i]))  # Subtract max for numerical stability
        softmax_matrix[i] = exp_row / np.sum(exp_row)
    print(softmax_matrix)

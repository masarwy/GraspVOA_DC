from utils.similarity_strategy import *
import matplotlib.pyplot as plt


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each row in matrix x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def mask_comp(image_a_file: str, image_b_file: str):
    mask1 = np.load(image_a_file)
    mask2 = np.load(image_b_file)
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


if __name__ == '__main__':
    sim_context = SimilarityContext()
    strategies = [StructureTermStrategy(), IoUStrategy(), ContourMatchStrategy()]

    sim_context.set_strategy(strategy=strategies[2])
    sim_matrices = []
    for sensor_id in range(6):
        sim_matrix = np.zeros((5, 5))
        for pose_id in range(5):
            real_image_file = f'../data/objects/ENDSTOP/img/lab/mdi_{sensor_id}_{pose_id}.npy'
            for pose_id_ in range(5):
                gen_image_file = f'../data/objects/ENDSTOP/img/gen/di_{sensor_id}_{pose_id_}.npy'
                sim_matrix[pose_id, pose_id_] = (
                    sim_context.compare_images(real_image_file, gen_image_file, a_is_real=True))
                mask_comp(real_image_file, gen_image_file)
        print(softmax(sim_matrix))
        sim_matrices.append(softmax(sim_matrix))
    print('done')

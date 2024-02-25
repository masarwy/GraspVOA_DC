from utils.similarity_strategy import *


def softmax(x):
    """Compute softmax values for each row in matrix x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


if __name__ == '__main__':
    sim_context = SimilarityContext()
    strategies = [StructureTermStrategy(), IoUStrategy(), ContourMatchStrategy()]

    sim_context.set_strategy(strategy=strategies[2])
    sim_matrices = []
    for sensor_id in range(6):
        sim_matrix = np.zeros((5, 5))
        for pose_id in range(5):
            real_image_file = f'../data/objects/ENDSTOP/img/lab/mdi_{sensor_id}_{pose_id}.npy'
            res = {}
            for pose_id_ in range(5):
                gen_image_file = f'../data/objects/ENDSTOP/img/gen/di_{sensor_id}_{pose_id_}.npy'
                sim_matrix[pose_id, pose_id_] = (
                    sim_context.compare_images(real_image_file, gen_image_file, a_is_real=True))
        sim_matrices.append(softmax(sim_matrix))
    print('done')

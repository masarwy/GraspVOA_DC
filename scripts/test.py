from utils.similarity_strategy import *


def dict_softmax(scores_dict: dict) -> dict:

    exp_values = np.exp(list(scores_dict.values()))
    sum_exp_values = np.sum(exp_values)
    softmax_probabilities = exp_values / sum_exp_values
    softmax_dict = dict(zip(scores_dict.keys(), softmax_probabilities))

    return softmax_dict


if __name__ == '__main__':

    sim_context = SimilarityContext()
    strategies = [StructureTermStrategy(), IoUStrategy(), ContourMatchStrategy()]

    sensor_id = 0
    real_image_file = f'../data/objects/ENDSTOP/img/lab/mdi_{sensor_id}_0.npy'

    gen_images_files = {}
    for pose_id in range(5):
        image_id = f'{sensor_id}_{pose_id}'
        gen_images_files[image_id] = '../data/objects/ENDSTOP/img/gen/di_' + image_id + '.npy'

    for strategy in strategies:
        similarity_scores = {}
        sim_context.set_strategy(strategy)
        for image_id in gen_images_files.keys():
            similarity_scores[image_id] = sim_context.compare_images(real_image_file, gen_images_files[image_id],
                                                                     a_is_real=True)
        print(sim_context.get_strategy_name(), ': ', dict_softmax(similarity_scores))



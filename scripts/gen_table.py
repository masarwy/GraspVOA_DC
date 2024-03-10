import glob

import yaml
import csv

from utils.similarity_strategy import *
from entities.pose_belief import BeliefSpaceModel
from utils.transform import Point3D
from scripts.test import dict_softmax
from scripts.compute_voa import compute_likelihood, gamma_bar

if __name__ == '__main__':
    sims = [StructureTermStrategy(), IoUStrategy(), ContourMatchStrategy()]
    sim_context = SimilarityContext()

    rows = []

    object_id = 'FLASK'
    obj_file = '../data/objects/' + object_id + '/object_.obj'
    obj_std_poses_file = '../data/objects/' + object_id + '/standard_poses.yaml'
    obj_sampled_poses_file = '../data/objects/' + object_id + '/sampled_poses.yaml'
    grasp_score_file = '../data/objects/' + object_id + '/grasp_score.yaml'
    poi = Point3D(-1.2, -1., 0)
    bm = BeliefSpaceModel(standard_poses_file=obj_std_poses_file, poi=poi)

    with open(grasp_score_file, 'r') as f:
        grasp_score = yaml.safe_load(f)

    with open(obj_sampled_poses_file, 'r') as file:
        data = yaml.safe_load(file)
        sampled_poses = data['poses']

    n_poses = len(sampled_poses)
    particles = np.empty((n_poses, 5), dtype=object)
    pred_likelihood = compute_likelihood(belief=bm, poses=sampled_poses, parts=particles)
    pred_soft_m = dict_softmax(pred_likelihood)
    pred_b = np.array([1 / n_poses] * n_poses)
    for i, sampled_pose in enumerate(sampled_poses):
        pred_b[i] = pred_soft_m[sampled_pose]
    init_x_star, init_exp_score, _, _, _, _, _ = gamma_bar(grasp_score=grasp_score, belief=pred_b)
    init_x_star_id = int(init_x_star[1]) - 1

    actual_likelihood = compute_likelihood(belief=bm, poses=sampled_poses, parts=particles)
    actual_soft_m = dict_softmax(actual_likelihood)

    actual_b = np.zeros_like(pred_b)
    for i, sampled_pose in enumerate(sampled_poses.keys()):
        actual_b[i] = actual_soft_m[sampled_pose]

    row = ['']
    for pose in sampled_poses.keys():
        row.extend([pose] + [''] * 16)
    row.append('VOA')
    rows.append(row)
    temp_row = ['init_x_star', 'init_score', 'init_exp_score', 'actual_x_star', 'actual_score', 'actual_exp_score',
                'pred_x_star', 'pred_grasp_score', 'pred_exp_score', 'full_info_g', 'full_info_score',
                'pose_prev_belief', 'pose_pred_belief', 'pose_actual_belief', 'init_b', 'pred_b', 'act_b'] * n_poses
    rows.append(['sensor conf.'] + temp_row)

    for sim in sims:
        print('_______________________________________________________________')
        sim_context.set_strategy(sim)
        print(sim_context.get_strategy_name())
        with open(grasp_score_file, 'r') as f:
            grasp_score = yaml.safe_load(f)

        with open(obj_sampled_poses_file, 'r') as file:
            data = yaml.safe_load(file)
            sampled_poses = data['poses']

        for sensor_id in range(6):
            file_pattern = f'../data/objects/ENDSTOP/img/gen/di_{sensor_id}_*.npy'
            image_files = sorted(glob.glob(file_pattern))

            if not image_files:
                raise ValueError("No image files found. Check the file pattern or directory.")

            n_images = len(image_files)
            similarity_matrix = np.zeros((n_images, n_images))

            for i in range(n_images):
                for j in range(i, n_images):
                    score = sim_context.compare_images(image_files[i], image_files[j], a_is_real=False)
                    similarity_matrix[i, j] = similarity_matrix[j, i] = score

            n_images = similarity_matrix.shape[0]
            softmax_matrix = np.zeros_like(similarity_matrix)
            for i in range(n_images):
                exp_row = np.exp(
                    similarity_matrix[i] - np.max(similarity_matrix[i]))
                softmax_matrix[i] = exp_row / np.sum(exp_row)

            row = [f'{sim_context.get_strategy_name()}: x{sensor_id}']
            voa = 0
            for i, pose_h in enumerate(sampled_poses.keys()):
                new_pred_b = np.zeros_like(pred_b)
                actual_sim_arr = np.zeros_like(actual_b)
                real_image_file = f'../data/objects/{object_id}/img/lab/mdi_{sensor_id}_{i}.npy'
                pose_prev_belief = pred_b[i]
                for j, pose_a in enumerate(sampled_poses.keys()):
                    new_pred_b[j] = pred_b[j] * softmax_matrix[i, j]
                    gen_image_file = f'../data/objects/{object_id}/img/gen/di_{sensor_id}_{j}.npy'
                    actual_sim_arr[j] = sim_context.compare_images(real_image_file, gen_image_file, a_is_real=True)

                new_pred_b /= new_pred_b.sum()
                pose_pred_belief = new_pred_b[i]
                pred_x_star, pred_exp_score, full_info_g, full_info_score, pred_grasp_score, init_score, _ = gamma_bar(
                    grasp_score=grasp_score,
                    belief=new_pred_b,
                    true_pose=i,
                    init_g=init_x_star_id)

                actual_exp_x = np.exp(actual_sim_arr - np.max(actual_sim_arr))
                actual_soft_arr = actual_exp_x / actual_exp_x.sum()
                new_actual_b = actual_b * actual_soft_arr
                new_actual_b /= new_actual_b.sum()

                pose_actual_belief = new_actual_b[i]
                actual_x_star, actual_exp_score, _, _, actual_score, _, _ = gamma_bar(grasp_score=grasp_score,
                                                                                   belief=new_actual_b, true_pose=i)

                print(pose_h, actual_x_star, actual_score, pred_x_star, pred_grasp_score)
                row.extend([init_x_star, init_score, init_exp_score, actual_x_star, actual_score, actual_exp_score,
                            pred_x_star, pred_grasp_score, pred_exp_score, full_info_g, full_info_score,
                            pose_prev_belief, pose_pred_belief, pose_actual_belief, pred_b, new_pred_b, new_actual_b])
                voa += (pred_grasp_score - init_score) * pred_b[i]
            row.append(voa)
            print(sensor_id, ' VOA: ', voa)
            rows.append(row)

    filename = f'../results/{object_id}/res.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

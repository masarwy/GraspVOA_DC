import glob

import numpy as np
import yaml
from typing import Tuple, Optional

from utils.similarity_strategy import *
from entities.pose_belief import BeliefSpaceModel
from utils.transform import Point3D
from scripts.test import dict_softmax


def gamma_bar(grasp_score: dict, belief: np.ndarray, true_pose: int = None, init_g: int = None) -> Tuple[
    str, float, Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]:
    dim1 = len(grasp_score)
    unique_keys = set(key for inner_dict in grasp_score.values() for key in inner_dict.keys())
    dim2 = len(unique_keys)
    w_table = np.zeros((dim2, dim1))
    table = w_table.copy()
    for p in range(dim2):
        for g in range(dim1):
            w_table[p, g] = grasp_score[f'G{g + 1}'][f'P{p + 1}'] * belief[p]
            table[p, g] = grasp_score[f'G{g + 1}'][f'P{p + 1}']
    column_sums = np.sum(w_table, axis=0)
    best_grasp = column_sums.argmax()
    true_best = None
    real_score = None
    init_score = None
    true_score = None
    exp_true_grasp = None
    if true_pose is not None:
        idx = table[true_pose, :].argmax()
        true_score = table[true_pose, idx]
        exp_true_grasp = column_sums[idx]
        true_best = f'G{idx + 1}'
        real_score = table[true_pose, best_grasp]
        init_score = table[true_pose, init_g]
    return f'G{best_grasp + 1}', column_sums[best_grasp], true_best, true_score, real_score, init_score, exp_true_grasp


def compute_best_grasp(grasp_score: dict, likelihood: dict) -> Tuple[str, float]:
    dim1 = len(grasp_score)
    unique_keys = set(key for inner_dict in grasp_score.values() for key in inner_dict.keys())
    dim2 = len(unique_keys)
    table = np.zeros((dim2, dim1))
    softmax = dict_softmax(likelihood)
    for p in range(dim2):
        for g in range(dim1):
            table[p, g] = grasp_score[f'G{g + 1}'][f'P{p + 1}'] * softmax[f'P{p + 1}']
    column_sums = np.sum(table, axis=0)
    best_grasp = column_sums.argmax()
    return f'G{best_grasp + 1}', column_sums[best_grasp]


def compute_likelihood(belief: BeliefSpaceModel, poses: dict, parts: np.ndarray) -> dict:
    likelihood = {}
    for i, pose in enumerate(poses.keys()):
        rep_pose = poses[pose]['our_rep']
        parts[i, 0] = category = rep_pose['category']
        parts[i, 1] = angle = rep_pose['angle']
        parts[i, 2] = x = rep_pose['x']
        parts[i, 3] = y = rep_pose['y']
        likelihood[pose] = belief.calculate_log_likelihood(category=category, angle=angle, x=x, y=y)
    return likelihood


if __name__ == '__main__':
    sims = [StructureTermStrategy(), IoUStrategy(), ContourMatchStrategy()]
    sim_context = SimilarityContext()
    for sim in sims:
        print('____________________________________________________________________')
        sim_context.set_strategy(sim)
        print(sim_context.get_strategy_name())
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

            n_images = similarity_matrix.shape[0]  # Assuming square matrix

            # Apply Softmax for each row
            softmax_matrix = np.zeros_like(similarity_matrix)

            for i in range(n_images):
                exp_row = np.exp(
                    similarity_matrix[i] - np.max(similarity_matrix[i]))  # Subtract max for numerical stability
                softmax_matrix[i] = exp_row / np.sum(exp_row)

            object_id = 'ENDSTOP'
            obj_file = '../data/objects/' + object_id + '/object.obj'
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
            likelihood = compute_likelihood(belief=bm, poses=sampled_poses, parts=particles)
            soft_m = dict_softmax(likelihood)

            b = np.array([1 / n_poses] * n_poses)
            for i, sampled_pose in enumerate(sampled_poses):
                b[i] = soft_m[sampled_pose]
            init_x_star, score, _, _, _, _, _ = gamma_bar(grasp_score=grasp_score, belief=b)
            print(init_x_star, score)
            print('_______________________________')
            voa = -score
            for i, pose_h in enumerate(sampled_poses.keys()):
                new_b = np.zeros_like(b)
                for j, pose_a in enumerate(sampled_poses.keys()):
                    new_b[j] = b[j] * softmax_matrix[i, j]
                new_b /= new_b.sum()
                x_star, score, actual, real_score, _, _, _ = gamma_bar(grasp_score=grasp_score, belief=new_b,
                                                                       true_pose=i)
                if i == 0:
                    print(x_star, score, actual, real_score)
                voa += score * b[i]
            print(sensor_id, ' VOA: ', voa)

        #     particles = np.empty((len(sampled_poses), 5), dtype=object)
        #     likelihood = compute_likelihood(belief=bm, poses=sampled_poses, parts=particles)
        #     soft_m = dict_softmax(likelihood)
        #
        #     best_grasp, score = compute_best_grasp(grasp_score=grasp_score, likelihood=soft_m)
        #     voa = -score
        #     for i, pose_h in enumerate(sampled_poses.keys()):
        #         # print(pose_h)
        #         for j, pose_a in enumerate(sampled_poses.keys()):
        #             particles[j, 4] = softmax_matrix[i, j]  # * np.exp(likelihood[pose_a])
        #             # print(particles[j, 4])
        #         bm.update_model(particles=particles)
        #         u_likelihood = compute_likelihood(belief=bm, poses=sampled_poses, parts=particles)
        #         u_soft_m = dict_softmax(u_likelihood)
        #         best_grasp, score = compute_best_grasp(grasp_score=grasp_score, likelihood=u_soft_m)
        #         voa += score * soft_m[pose_h]
        #         bm.reset()
        #     print(sensor_id, ' VOA: ', voa)
        # print('__________________')

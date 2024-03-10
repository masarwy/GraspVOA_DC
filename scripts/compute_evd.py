import yaml
import glob

from utils.similarity_strategy import *
from entities.pose_belief import BeliefSpaceModel
from utils.transform import Point3D
from scripts.test import dict_softmax
from scripts.compute_voa import compute_likelihood, compute_best_grasp, gamma_bar

if __name__ == '__main__':
    sims = [StructureTermStrategy(), IoUStrategy(), ContourMatchStrategy()]
    sim_context = SimilarityContext()

    object_id = 'ENDSTOP'
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

    for sim in sims:
        print('_______________________________________________________________')
        sim_context.set_strategy(sim)
        print(sim_context.get_strategy_name())
        for sensor_id in range(6):
            file_pattern = f'../data/objects/ENDSTOP/img/lab/mdi_{sensor_id}_*.npy'
            image_files = sorted(glob.glob(file_pattern))

            if not image_files:
                raise ValueError("No image files found. Check the file pattern or directory.")

            n_images = len(image_files)

            n_poses = len(sampled_poses)
            particles = np.empty((n_poses, 5), dtype=object)
            likelihood = compute_likelihood(belief=bm, poses=sampled_poses, parts=particles)
            soft_m = dict_softmax(likelihood)

            b = np.array([0] * n_poses)
            for i, sampled_pose in enumerate(sampled_poses):
                b[i] = soft_m[sampled_pose]

            init_x_star, score, _, _, _, _, _ = gamma_bar(grasp_score=grasp_score, belief=b)
            evd = -score
            print(init_x_star, score)
            print('_______________________________')
            for i, pose in enumerate(sampled_poses):
                sim_arr = np.zeros_like(b)
                real_image_file = f'../data/objects/{object_id}/img/lab/mdi_{sensor_id}_{i}.npy'
                for j, pose_ in enumerate(sampled_poses):
                    gen_image_file = f'../data/objects/{object_id}/img/gen/di_{sensor_id}_{j}.npy'
                    sim_arr[j] = sim_context.compare_images(real_image_file, gen_image_file, a_is_real=True)
                exp_x = np.exp(sim_arr - np.max(sim_arr))
                soft_arr = exp_x / exp_x.sum()
                new_b = b * soft_arr
                new_b /= new_b.sum()
                x_star, score, true_x, _, _, _, _ = gamma_bar(grasp_score=grasp_score, belief=new_b, true_pose=i)
                if i == 0:
                    print(x_star, score, true_x)
                evd += score * b[i]
            print(sensor_id, ' EVD: ', evd)

            # particles = np.empty((len(sampled_poses), 5), dtype=object)
            # likelihood = compute_likelihood(belief=bm, poses=sampled_poses, parts=particles)
            # softm = dict_softmax(likelihood)
            #
            # evd = 0
            # for i, pose in enumerate(sampled_poses):
            #     best_grasp = compute_best_grasp(grasp_score=grasp_score, likelihood=softm)
            #     evd -= grasp_score[best_grasp][pose] * softm[pose]
            #     real_image_file = f'../data/objects/ENDSTOP/img/lab/mdi_{sensor_id}_{i}.npy'
            #     res = {}
            #     for j, pose_ in enumerate(sampled_poses):
            #         gen_image_file = f'../data/objects/ENDSTOP/img/gen/di_{sensor_id}_{j}.npy'
            #         particles[j, 4] = sim_context.compare_images(real_image_file, gen_image_file,
            #                                                      a_is_real=True) * np.exp(likelihood[pose_])
            #     bm.update_model(particles=particles)
            #     likelihood = compute_likelihood(belief=bm, poses=sampled_poses, parts=particles)
            #     u_softm = dict_softmax(likelihood)
            #     best_grasp = compute_best_grasp(grasp_score=grasp_score, likelihood=u_softm)
            #     evd += grasp_score[best_grasp][pose] * softm[pose]
            #     bm.reset()
            #     likelihood = compute_likelihood(belief=bm, poses=sampled_poses, parts=particles)
            # print(sensor_id, ': ', evd)

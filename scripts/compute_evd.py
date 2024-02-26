import yaml
import glob

from utils.similarity_strategy import *
from entities.pose_belief import BeliefSpaceModel
from utils.transform import Point3D
from scripts.test import dict_softmax
from scripts.compute_voa import compute_likelihood, compute_best_grasp

if __name__ == '__main__':
    sims = [StructureTermStrategy(), IoUStrategy(), ContourMatchStrategy()]
    sim_context = SimilarityContext()

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

    for sim in sims:
        sim_context.set_strategy(sim)
        print(sim_context.get_strategy_name())
        for sensor_id in range(6):
            print(sensor_id)
            file_pattern = f'../data/objects/ENDSTOP/img/lab/mdi_{sensor_id}_*.npy'
            image_files = sorted(glob.glob(file_pattern))

            if not image_files:
                raise ValueError("No image files found. Check the file pattern or directory.")

            n_images = len(image_files)

            particles = np.empty((len(sampled_poses), 5), dtype=object)
            likelihood = compute_likelihood(belief=bm, poses=sampled_poses, parts=particles)

            evd = 0
            for i, pose in enumerate(sampled_poses):
                real_image_file = f'../data/objects/ENDSTOP/img/lab/mdi_{sensor_id}_{i}.npy'
                res = {}
                for j, pose_ in enumerate(sampled_poses):
                    gen_image_file = f'../data/objects/ENDSTOP/img/gen/di_{sensor_id}_{j}.npy'
                    particles[j, 4] = sim_context.compare_images(real_image_file, gen_image_file,
                                                                 a_is_real=True) * np.exp(likelihood[pose_])
                bm.update_model(particles=particles)
                likelihood = compute_likelihood(belief=bm, poses=sampled_poses, parts=particles)
                softm = dict_softmax(likelihood)
                # print(softm)
                best_grasp = compute_best_grasp(grasp_score=grasp_score, likelihood=softm)
                evd += grasp_score[best_grasp][pose] * softm[pose]
                bm.reset()
                likelihood = compute_likelihood(belief=bm, poses=sampled_poses, parts=particles)
            print(sensor_id, ': ', evd)

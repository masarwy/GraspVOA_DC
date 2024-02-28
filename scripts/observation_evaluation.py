import numpy as np

from utils.sensor_conf import read_sensor_configs

if __name__ == '__main__':
    sensor_poses_file = '../data/poses_and_joints.yaml'
    sensor_p_q = read_sensor_configs(sensor_poses_file)
    poi = np.array([-1.2, -1., 0.])
    print(sensor_p_q)
    for i, conf in enumerate(sensor_p_q.keys()):
        avg = 0
        max_req = 0
        pos = np.array([sensor_p_q[conf]['pose'].x, sensor_p_q[conf]['pose'].y, sensor_p_q[conf]['pose'].z])
        dist = np.linalg.norm(poi - pos)
        for pose_id in range(6):
            mask1 = np.load(f'../data/objects/ENDSTOP/img/lab/mdi_{i}_{pose_id}.npy')
            mask2 = np.load(f'../data/objects/ENDSTOP/img/gen/di_{i}_{pose_id}.npy')
            mask1 = mask1 != 0.
            mask2 = mask2 != 0.
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            iou = intersection / union if union != 0 else 0
            max_req = max(max_req, iou)
            print(mask1.sum(), mask2.sum(), iou)
            avg += iou / 6
        print('avg: ', avg)
        print('dist: ', dist)
        print('max_req: ', max_req)
        print('_______________________________________________________________________________')

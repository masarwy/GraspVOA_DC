import cv2
import numpy as np

if __name__ == "__main__":
    sen_id = 0
    obj_pose_id = 1
    binary_mask_path = f'../data/objects/ENDSTOP/img/lab/mask_{sen_id}_{obj_pose_id}.png'
    depth_image_path = f'../data/objects/ENDSTOP/img/lab/di_{sen_id}_{obj_pose_id}.npy'
    save_path = f'../data/objects/ENDSTOP/img/lab/mdi_{sen_id}_{obj_pose_id}.npy'

    binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_UNCHANGED)
    depth_image = np.load(depth_image_path)

    binary_mask = binary_mask.astype(np.uint16)
    resulting_image = depth_image * binary_mask
    np.save(save_path, resulting_image)

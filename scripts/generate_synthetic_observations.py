import numpy as np
import yaml
from PIL import Image

from entities.pose_belief import BeliefSpaceModel
from utils.transform import Transform, Point3D, Pose
from entities.object import Object
from entities.render import Render
from utils.common_transforms import CameraPoseExtractor
from utils.sensor_conf import read_sensor_configs

import trimesh

if __name__ == '__main__':
    render_scene = False

    object_id = 'ENDSTOP'
    obj_file = '../data/objects/' + object_id + '/object.obj'
    obj_std_poses_file = '../data/objects/' + object_id + '/standard_poses.yaml'
    obj_sampled_poses_file = '../data/objects/' + object_id + '/sampled_poses.yaml'
    sensor_poses_file = '../data/poses_and_joints.yaml'
    poi = Point3D(-1.2, -1., 0)

    sensor_p_q = read_sensor_configs(sensor_poses_file)

    with open(obj_std_poses_file, 'r') as file:
        data = yaml.safe_load(file)
        standard_poses = data['standard_poses']

    bm = BeliefSpaceModel(obj_std_poses_file, poi=poi)

    with open(obj_sampled_poses_file, 'r') as file:
        data = yaml.safe_load(file)
        sampled_poses = data['poses']

    transforms = []
    for pose in sampled_poses.keys():
        x, y, z = sampled_poses[pose]['position_offset']
        rx, ry, rz = sampled_poses[pose]['orientation']
        transforms.append(Transform.from_pose_xyz(Pose(x, y, z, rx, ry, rz)))

    camera_params_file = '../data/camera_params.yaml'
    with open(camera_params_file, 'r') as file:
        config = yaml.safe_load(file)
    camera_params = config['camera_params']
    yfov = np.radians(camera_params['fov_vertical_rad'])
    aspect_ratio = camera_params['image_width'] / camera_params['image_height']

    obj = Object(obj_file)
    render = Render(yfov=yfov, aspectRatio=aspect_ratio, width=camera_params['image_width'],
                    height=camera_params['image_height'])

    camera_in_ee = Transform(rotation=np.eye(3), translation=np.array([-0.0075, -0.105, 0.0395]), from_frame='EE',
                             to_frame='camera')
    camera_in_world_calc = CameraPoseExtractor(camera_in_ee=camera_in_ee)

    # render empty scenes
    for sensor_id in sensor_p_q.keys():
        sen_id = int(sensor_id[-1])
        pose_rv = sensor_p_q[sensor_id]['pose']
        j_state = sensor_p_q[sensor_id]['joints']
        camera_pose = camera_in_world_calc(ee_in_world=Transform.from_rv(pose_rv)).adjust_to_look_at_format()
        _, depth = render.render_empty_scene(camera_pose=camera_pose.get_transformation_matrix())
        np.save(f'../data/empty_scene/gen/di_{sen_id}.npy', depth)

        image_min = depth.min()
        image_max = depth.max()

        # Normalize to 0-1 range
        normalized_image = (depth - image_min) / (image_max - image_min)

        # Scale to 0-255 and convert to uint8
        scaled_image = (normalized_image * 255).astype(np.uint8)
        image = Image.fromarray(scaled_image)
        image.save(f'../data/empty_scene/gen/di_{sen_id}.png')

    counter = 0
    for t in transforms:
        obj.set_transform(t)
        if render_scene:
            scene = trimesh.Scene()
            scene.add_geometry(obj.get_mesh())
            axis = trimesh.creation.axis(axis_length=1.0, origin_size=0.01)
            scene.add_geometry(axis)
            scene.show()
        for sensor_id in sensor_p_q.keys():
            pose_rv = sensor_p_q[sensor_id]['pose']
            j_state = sensor_p_q[sensor_id]['joints']
            camera_pose = camera_in_world_calc(ee_in_world=Transform.from_rv(pose_rv)).adjust_to_look_at_format()
            _, depth, mask = render.render_scene(mesh=obj.get_mesh(),
                                                 camera_pose=camera_pose.get_transformation_matrix(), add_plane=False)

            sen_id = int(sensor_id[-1])

            np.save('../data/objects/' + object_id + f'/img/gen/di_{sen_id}_{counter}.npy', depth)

            image_min = depth.min()
            image_max = depth.max()

            # Normalize to 0-1 range
            normalized_image = (depth - image_min) / (image_max - image_min)

            # Scale to 0-255 and convert to uint8
            scaled_image = (normalized_image * 255).astype(np.uint8)
            image = Image.fromarray(scaled_image)
            image.save('../data/objects/' + object_id + f'/img/gen/di_{sen_id}_{counter}.png')
        counter += 1

    print('done')

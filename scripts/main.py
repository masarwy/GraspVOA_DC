import numpy as np
import yaml

from pose_belief import BeliefSpaceModel
from transform import Transform, Point3D
from object import Object
from render import Render
from common_transforms import CameraPoseExtractor
from sensor_conf import read_sensor_configs

import trimesh

if __name__ == '__main__':
    render_scene = False

    object_id = 'ENDSTOP'
    obj_file = '../data/objects/' + object_id + '/object.obj'
    obj_poses_file = '../data/objects/' + object_id + '/standard_poses.yaml'
    sensor_poses_file = '../data/poses_and_joints.yaml'
    poi = Point3D(-1.2, -1., 0)

    sensor_p_q = read_sensor_configs(sensor_poses_file)

    bm = BeliefSpaceModel(obj_poses_file, poi=poi)
    particles = bm.sample_particles(n_particles=10)
    transforms = []
    for category, angle, x, y, _ in particles:
        transforms.append(Transform.from_pose_zyx(bm.particle_to_6d_pose(category, float(angle), float(x), float(y))))

    camera_params_file = '../data/camera_params.yaml'
    with open(camera_params_file, 'r') as file:
        config = yaml.safe_load(file)
    camera_params = config['camera_params']
    yfov = np.radians(camera_params['fov_vertical_rad'])
    aspect_ratio = camera_params['image_width'] / camera_params['image_height']

    obj = Object(obj_file)
    render = Render(yfov=yfov, aspectRatio=aspect_ratio)

    workspace_in_world = Transform(rotation=np.eye(3), translation=np.array([-0.7, 0, 0]), from_frame='world',
                                   to_frame='workspace')
    camera_in_ee = Transform(rotation=np.eye(3), translation=np.array([0, -0.105, 0.0395]), from_frame='EE',
                             to_frame='camera')
    camera_in_world_calc = CameraPoseExtractor(camera_in_ee=camera_in_ee)

    masks = []
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
            _, _, mask = render.render_scene(mesh=obj.get_mesh(), camera_pose=camera_pose.get_transformation_matrix())
            masks.append(mask)

    print('done')

    # if render_scene:
    #     scene = trimesh.Scene()
    #     scene.add_geometry(obj.get_mesh())
    #     # Create an axis object
    #     axis = trimesh.creation.axis(axis_length=1.0, origin_size=0.01)
    #
    #     # Add the axis object to the scene
    #     scene.add_geometry(axis)
    #     scene.show()

    # look_at_generator = camera.generate_camera_poses(n_samples_theta=4, n_samples_phi=3)
    #
    # counter = 0
    # for look_at in look_at_generator:
    #     _, depth, mask = render.render_scene(mesh=obj.get_mesh(), camera_pose=look_at.get_transformation_matrix())
    #
    #     print(ee_extractor(camera_in_workspace=look_at.adjust_for_camera_pose()).to_pose_zyx())
    #     print(look_at.adjust_for_camera_pose().to_pose_zyx())
    #     print('________________________________')
    #     image_min = depth.min()
    #     image_max = depth.max()
    #
    #     # Normalize to 0-1 range
    #     normalized_image = (depth - image_min) / (image_max - image_min)
    #
    #     # Scale to 0-255 and convert to uint8
    #     scaled_image = (normalized_image * 255).astype(np.uint8)
    #     image = Image.fromarray(scaled_image)
    #     image.save('../data/objects/' + object_id + f'/img/di_{counter}.png')
    #     counter += 1

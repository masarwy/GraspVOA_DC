import numpy as np
from PIL import Image
import yaml

from pose_belief import BeliefSpaceModel
from transform import Transform, Point3D
from object import Object
from render import Render
from camera import Camera
from ee_pose_extractor import EEPoseExtractor
from sensor_configs_sampler import heuristic_driven_sampling

import trimesh

if __name__ == '__main__':
    render_scene = True

    object_id = 'ENDSTOP'
    obj_file = '../data/objects/' + object_id + '/object.obj'
    poses_file = '../data/objects/' + object_id + '/standard_poses.yaml'

    bm = BeliefSpaceModel(poses_file)
    particles = bm.sample_particles(n_particles=10)
    transforms = []
    for category, angle, x, y, _ in particles:
        transforms.append(Transform.from_pose(bm.particle_to_6d_pose(category, float(angle), float(x), float(y))))

    camera_params_file = '../data/camera_params.yaml'
    with open(camera_params_file, 'r') as file:
        config = yaml.safe_load(file)
    camera_params = config['camera_params']
    yfov = np.radians(camera_params['fov_vertical_rad'])
    aspect_ratio = camera_params['image_width'] / camera_params['image_height']

    obj = Object(obj_file)
    camera = Camera(radius=0.3)
    render = Render(yfov=yfov, aspectRatio=aspect_ratio)

    workspace_in_world = Transform(rotation=np.eye(3), translation=np.array([-0.7, 0, 0]), from_frame='world',
                                   to_frame='workspace')
    camera_in_ee = Transform(rotation=np.eye(3), translation=np.array([0, -0.105, 0.0395]), from_frame='EE',
                             to_frame='camera')
    ee_extractor = EEPoseExtractor(workspace_in_world=workspace_in_world, camera_in_ee=camera_in_ee)

    for t in transforms:
        obj.set_transform(t)
        scene = trimesh.Scene()
        scene.add_geometry(obj.get_mesh())
        # Create an axis object
        axis = trimesh.creation.axis(axis_length=1.0, origin_size=0.01)

        # Add the axis object to the scene
        scene.add_geometry(axis)
        scene.show()

    _, _, mask = render.render_scene(mesh=obj.get_mesh(),
                                     camera_pose=camera.create_camera_pose_from_x().get_transformation_matrix())
    joint_limits = np.zeros((6, 2))
    joint_limits[:, 1] = np.array([360] * 6)
    X_s = heuristic_driven_sampling(camera_in_ee=camera_in_ee, camera_params=camera_params, joint_limits=joint_limits,
                                    poi=Point3D(50, 50, 50))

    if render_scene:
        scene = trimesh.Scene()
        scene.add_geometry(obj.get_mesh())
        # Create an axis object
        axis = trimesh.creation.axis(axis_length=1.0, origin_size=0.01)

        # Add the axis object to the scene
        scene.add_geometry(axis)
        scene.show()

    look_at_generator = camera.generate_camera_poses(n_samples_theta=4, n_samples_phi=3)

    counter = 0
    for look_at in look_at_generator:
        _, depth, mask = render.render_scene(mesh=obj.get_mesh(), camera_pose=look_at.get_transformation_matrix())

        print(ee_extractor(camera_in_workspace=look_at.adjust_for_camera_pose()).to_pose())
        print(look_at.adjust_for_camera_pose().to_pose())
        print('________________________________')
        image_min = depth.min()
        image_max = depth.max()

        # Normalize to 0-1 range
        normalized_image = (depth - image_min) / (image_max - image_min)

        # Scale to 0-255 and convert to uint8
        scaled_image = (normalized_image * 255).astype(np.uint8)
        image = Image.fromarray(scaled_image)
        image.save('../data/objects/' + object_id + f'/img/di_{counter}.png')
        counter += 1

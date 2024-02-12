import numpy as np
from PIL import Image

from object import Object
from render import Render
from camera import Camera

import trimesh

if __name__ == '__main__':
    obj_file = 'object.obj'
    render_scene = False

    obj = Object(obj_file)
    camera = Camera(radius=0.3)
    render = Render(yfov=np.pi / 3.0)

    _, _, mask = render.render_scene(mesh=obj.get_mesh(),
                                     camera_pose=camera.create_camera_pose_from_x().get_transformation_matrix())

    if render_scene:
        scene = trimesh.Scene()
        scene.add_geometry(obj.get_mesh())
        # Create an axis object
        axis = trimesh.creation.axis(axis_length=1.0, origin_size=0.01)

        # Add the axis object to the scene
        scene.add_geometry(axis)
        scene.show()

    pose_generator = camera.generate_camera_poses(n_samples_theta=4, n_samples_phi=3)

    counter = 0
    for pose in pose_generator:
        _, depth, mask = render.render_scene(mesh=obj.get_mesh(), camera_pose=pose.get_transformation_matrix())
        image_min = depth.min()
        image_max = depth.max()

        # Normalize to 0-1 range
        normalized_image = (depth - image_min) / (image_max - image_min)

        # Scale to 0-255 and convert to uint8
        scaled_image = (normalized_image * 255).astype(np.uint8)
        image = Image.fromarray(scaled_image)
        image.save(f'../data/img/di_{counter}.png')
        counter += 1

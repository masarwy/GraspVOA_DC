import numpy as np

from object import Object
from render import Render
from camera import Camera

import matplotlib.pyplot as plt
import trimesh

if __name__ == '__main__':
    obj_file = 'object.obj'

    obj = Object(obj_file)
    camera = Camera(radius=0.5)
    render = Render(yfov=np.pi / 3.0)

    _, _, mask = render.render_scene(mesh=obj.get_mesh(),
                                     camera_pose=camera.create_camera_pose_from_x().get_transformation_matrix())
    print(mask)

    scene = trimesh.Scene()
    scene.add_geometry(obj.get_mesh())
    # Create an axis object
    axis = trimesh.creation.axis(axis_length=1.0, origin_size=0.01)

    # Add the axis object to the scene
    scene.add_geometry(axis)
    scene.show()

    # pose_generator = camera.generate_camera_poses(n_samples_theta=4, n_samples_phi=3)
    #
    # for pose in pose_generator:
    #     _, _, mask = render.render_scene(mesh=obj.get_mesh(), camera_pose=pose.get_transformation_matrix())
    #     print(mask)

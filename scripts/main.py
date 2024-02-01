import numpy as np

from object import Object
from render import Render
from camera import Camera

if __name__ == '__main__':
    obj_file = 'object.obj'

    obj = Object(obj_file)
    camera = Camera(radius=0.5)
    render = Render(yfov=np.pi / 3.0)

    pose_generator = camera.generate_poses(n_theta=4, n_phi=3)

    for pose in pose_generator:
        _, _, mask = render.render_scene(mesh=obj.get_mesh(), camera_pose=pose.get_transformation_matrix())
        print(mask)

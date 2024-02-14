import pyrender
import numpy as np


class Render:
    def __init__(self, yfov: float, aspectRatio: float):
        self.camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspectRatio)

    # returns color, depth, mask
    def render_scene(self, mesh, camera_pose):
        self.render_mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene()
        scene.add(self.render_mesh)
        scene.add(self.render_mesh)
        scene.add(self.camera, pose=camera_pose)
        renderer = pyrender.OffscreenRenderer(640, 480)
        color, depth = renderer.render(scene)
        mask = 255 - np.all(color != [0, 0, 0], axis=-1).astype(np.uint8) * 255
        renderer.delete()
        return color, depth, mask

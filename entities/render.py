import pyrender
import numpy as np
import trimesh


class Render:
    def __init__(self, yfov: float, aspectRatio: float, width: int, height: int):
        self.camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspectRatio)
        self.width = width
        self.height = height

    # returns color, depth, mask
    def render_scene(self, mesh, camera_pose, add_plane=False):
        render_mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene()
        scene.add(render_mesh)

        scene.add(self.camera, pose=camera_pose)
        renderer = pyrender.OffscreenRenderer(self.width, self.height)
        color, depth = renderer.render(scene)
        mask = 255 - np.all(color != [0, 0, 0], axis=-1).astype(np.uint8) * 255

        if add_plane:
            plane = trimesh.creation.box(extents=(10, 10, 0.01))
            plane_mesh = pyrender.Mesh.from_trimesh(plane)
            scene.add(plane_mesh)
            renderer = pyrender.OffscreenRenderer(self.width, self.height)
            color, depth = renderer.render(scene)
        renderer.delete()
        return color, depth, mask

    def render_empty_scene(self, camera_pose):
        scene = pyrender.Scene()
        scene.add(self.camera, pose=camera_pose)
        plane = trimesh.creation.box(extents=(10, 10, 0.01))
        plane_mesh = pyrender.Mesh.from_trimesh(plane)
        scene.add(plane_mesh)
        renderer = pyrender.OffscreenRenderer(self.width, self.height)
        color, depth = renderer.render(scene)
        renderer.delete()
        return color, depth

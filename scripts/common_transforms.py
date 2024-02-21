from transform import Transform


class EEPoseExtractor:
    def __init__(self, workspace_in_world: Transform, camera_in_ee: Transform):
        self.workspace_in_world = workspace_in_world
        self.camera_in_ee = camera_in_ee

    def __call__(self, camera_in_workspace: Transform) -> Transform:
        ee_in_workspace = camera_in_workspace.compose(self.camera_in_ee.inverse())
        ee_in_world = self.workspace_in_world.compose(ee_in_workspace)
        return ee_in_world


class CameraPoseExtractor:
    def __init__(self, camera_in_ee: Transform):
        self.camera_in_ee = camera_in_ee

    def __call__(self, ee_in_world: Transform) -> Transform:
        camera_in_world = ee_in_world.compose(self.camera_in_ee)
        return camera_in_world

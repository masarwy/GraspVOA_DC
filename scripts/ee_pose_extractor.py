from transform import Transform


class EEPoseExtractor:
    def __init__(self, workspace_in_world: Transform, camera_in_ee: Transform):
        self.workspace_in_world = workspace_in_world
        self.camera_in_ee = camera_in_ee

    def extract_pose(self, camera_in_workspace: Transform):
        ee_in_workspace = camera_in_workspace.compose(self.camera_in_ee.inverse())
        ee_in_world = self.workspace_in_world.compose(ee_in_workspace)
        return ee_in_world

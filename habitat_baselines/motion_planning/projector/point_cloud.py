import numpy as np
import torch

from habitat_baselines.motion_planning.projector.core import ProjectorUtils


class PointCloud(ProjectorUtils):
    """
    Unprojects 2D depth pixels in 3D
    """

    def __init__(
        self,
        vfov,
        batch_size,
        feature_map_height,
        feature_map_width,
        world_shift_origin,
        z_clip_threshold,
        device=None,
    ):
        """Init function

        Args:
            vfov (float): Vertical Field of View
            batch_size (float)
            feature_map_height (int): height of image
            feature_map_width (int): width of image
            world_shift_origin (float, float, float): (x, y, z) shift apply to position the map in the world coordinate system.
            z_clip_threshold (float): in meters. Pixels above camera height + z_clip_threshold will be ignored. (mainly ceiling pixels)
            device (torch.device, optional): Defaults to torch.device('cuda').
        """
        if device is None:
            device = torch.device("cuda")

        ProjectorUtils.__init__(
            self,
            vfov,
            batch_size,
            feature_map_height,
            feature_map_width,
            1,
            1,
            1,
            world_shift_origin,
            z_clip_threshold,
            device,
        )

        self.vfov = vfov
        self.batch_size = batch_size
        self.fmh = feature_map_height
        self.fmw = feature_map_width
        self.world_shift_origin = world_shift_origin
        self.z_clip_threshold = z_clip_threshold
        self.device = device

    def forward(self, depth, T, obs_per_map=1):
        """Forward Function

        Args:
            depth (torch.FloatTensor): Depth image
            T (torch.FloatTensor): camera-to-world transformation matrix
                                        (inverse of extrinsic matrix)
            obs_per_map (int): obs_per_map images are projected to the same map

        Returns:
            mask (torch.FloatTensor): mask of outliers. Mainly when no depth is present.
            point cloud (torch.FloatTensor)

        """

        assert depth.shape[2] == self.fmh
        assert depth.shape[3] == self.fmw

        depth = depth[:, 0, :, :]

        # -- filter out the semantic classes with depth == 0. Those sem_classes map to the agent
        # itself .. and thus are considered outliers
        no_depth_mask = depth == 0

        # Feature mappings in the world coordinate system where origin is somewhere but not camera
        # # GEO:
        # shape: features_to_world (N, features_height, features_width, 3)
        point_cloud = self.pixel_to_world_mapping(depth, T)

        return point_cloud, no_depth_mask


if __name__ == "__main__":
    from core import _transform3D
    from scipy.spatial.transform import Rotation as R

    from habitat import get_config
    from habitat.sims import make_sim

    house = "17DRP5sb8fy"
    scene = "../data/mp3d/{}/{}.glb".format(house, house)
    config = get_config()
    config.defrost()
    config.SIMULATOR.SCENE = scene
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR"]
    config.freeze()

    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)

    sim.reset()

    vfov = 67.5
    world_shift = torch.FloatTensor([0, 0, 0])
    projector = PointCloud(
        vfov,
        1,
        480,
        640,
        world_shift,
        0.5,
        device=torch.device("cpu"),
    )

    ags = sim.get_agent_state()
    pos = ags.sensor_states["depth"].position
    rot = ags.sensor_states["depth"].rotation
    rot = np.array([rot.x, rot.y, rot.z, rot.w])
    r = R.from_quat(rot)
    elevation, heading, bank = r.as_rotvec()

    xyzhe = np.array(
        [[pos[0], pos[1], pos[2], heading, elevation + np.pi]]
    )  # -- in Habitat y is up
    xyzhe = torch.FloatTensor(xyzhe)
    T = _transform3D(xyzhe)

    # -- depth for projection
    depth = sim.render(mode="depth")
    depth = depth[:, :, 0]
    depth = depth.astype(np.float32)
    depth *= 10.0
    depth_var = torch.FloatTensor(depth).unsqueeze(0).unsqueeze(0)

    pc, mask_outliers = projector.forward(depth_var, T)

    pc = pc[~mask_outliers]

    import matplotlib.pyplot as plt
    import numpy as np

    pc = pc.numpy()

    pc = pc[0:-1:100, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    ax.scatter(x, y, z)

    plt.show()

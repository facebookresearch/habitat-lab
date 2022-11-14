import torch
import numpy as np
from typing import List

from agent.mapping.metric.map_utils import MapSizeParameters, init_map_and_pose_for_env
from agent.perception.segmentation.lseg.modules.models.lseg_net import LSegEncDecNet


class VisionLanguage2DSemanticMapState:
    """
    This class holds a dense 2D semantic map with vision-language features,
    the global and local map and sensor pose, as well as the agent's
    current goal in the local map.

    Map proposed in:
    Visual Language Maps for Robot Navigation
    https://arxiv.org/pdf/2210.05714.pdf
    https://github.com/vlmaps/vlmaps
    """

    def __init__(
        self,
        device: torch.device,
        num_environments: int,
        lseg_features_dim: int,
        map_resolution: int,
        map_size_cm: int,
        global_downscaling: int,
    ):
        """
        Arguments:
            device: torch device on which to store map state
            num_environments: number of parallel maps (always 1 in real-world but
             multiple in simulation)
            lseg_features_dim: number of dimensions of vision-language (CLIP) features
            map_resolution: size of map bins (in centimeters)
            map_size_cm: global map size (in centimetres)
            global_downscaling: ratio of global over local map
        """
        self.device = device
        self.num_environments = num_environments
        self.lseg_features_dim = lseg_features_dim

        self.map_size_parameters = MapSizeParameters(
            map_resolution, map_size_cm, global_downscaling
        )
        self.resolution = map_resolution
        self.global_map_size_cm = map_size_cm
        self.global_downscaling = global_downscaling
        self.local_map_size_cm = self.global_map_size_cm // self.global_downscaling
        self.global_map_size = self.global_map_size_cm // self.resolution
        self.local_map_size = self.local_map_size_cm // self.resolution

        # Map consists of multiple channels containing the following:
        # 0: Obstacle Map
        # 1: Explored Area
        # 2: Current Agent Location
        # 3: Past Agent Locations
        # 4: Number of Cell Updates
        # 5, 6, .., lseg_features_dim + 5: LSeg CLIP features
        num_channels = lseg_features_dim + 5

        self.global_map = torch.zeros(
            self.num_environments,
            num_channels,
            self.global_map_size,
            self.global_map_size,
            device=self.device,
        )
        self.local_map = torch.zeros(
            self.num_environments,
            num_channels,
            self.local_map_size,
            self.local_map_size,
            device=self.device,
        )

        # Global and local (x, y, o) sensor pose
        self.global_pose = torch.zeros(self.num_environments, 3, device=self.device)
        self.local_pose = torch.zeros(self.num_environments, 3, device=self.device)

        # Origin of local map (3rd dimension stays 0)
        self.origins = torch.zeros(self.num_environments, 3, device=self.device)

        # Local map boundaries
        self.lmb = torch.zeros(
            self.num_environments, 4, dtype=torch.int32, device=self.device
        )

        # Binary map encoding agent high-level goal
        self.goal_map = np.zeros(
            (self.num_environments, self.local_map_size, self.local_map_size)
        )

    def init_map_and_pose(self):
        """Initialize global and local map and sensor pose variables."""
        for e in range(self.num_environments):
            self.init_map_and_pose_for_env(e)

    def init_map_and_pose_for_env(self, e: int):
        """Initialize global and local map and sensor pose variables for
        a specific environment.
        """
        init_map_and_pose_for_env(
            e,
            self.local_map,
            self.global_map,
            self.local_pose,
            self.global_pose,
            self.lmb,
            self.origins,
            self.map_size_parameters,
        )
        self.goal_map[e] *= 0.0

    def update_global_goal_for_env(self, e: int, goal_map: np.ndarray):
        """Update global goal for a specific environment with the goal action chosen
        by the policy.

        Arguments:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
        """
        self.goal_map[e] = goal_map

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_obstacle_map(self, e) -> np.ndarray:
        """Get local obstacle map for an environment."""
        return np.copy(self.local_map[e, 0, :, :].cpu().float().numpy())

    def get_explored_map(self, e) -> np.ndarray:
        """Get local explored map for an environment."""
        return np.copy(self.local_map[e, 1, :, :].cpu().float().numpy())

    def get_visited_map(self, e) -> np.ndarray:
        """Get local visited map for an environment."""
        return np.copy(self.local_map[e, 3, :, :].cpu().float().numpy())

    def get_semantic_map(self, e, lseg: LSegEncDecNet, labels: List[str]) -> np.ndarray:
        """Get local map of semantic categories for an environment - decode CLIP
        features to label set."""
        assert labels[-1] == "other"
        one_hot_categories, _ = lseg.decode(
            self.local_map[[e], 5:, :, :], labels
        )
        one_hot_categories = one_hot_categories.squeeze(0)
        empty_mask = self.local_map[e, 5:].sum(0) == 0
        one_hot_categories[empty_mask, :] = 0
        one_hot_categories = one_hot_categories.cpu().float().numpy()
        one_hot_categories[:, :, -1] = 1e-5  # Last category is "other"
        semantic_map = one_hot_categories.argmax(-1)
        return semantic_map

    def get_planner_pose_inputs(self, e) -> np.ndarray:
        """Get local planner pose inputs for an environment.

        Returns:
            planner_pose_inputs with 7 dimensions:
             1-3: Global pose
             4-7: Local map boundaries
        """
        return (
            torch.cat([self.local_pose[e] + self.origins[e], self.lmb[e]])
            .cpu()
            .float()
            .numpy()
        )

    def get_goal_map(self, e) -> np.ndarray:
        """Get binary goal map encoding current global goal for an
        environment."""
        return self.goal_map[e]

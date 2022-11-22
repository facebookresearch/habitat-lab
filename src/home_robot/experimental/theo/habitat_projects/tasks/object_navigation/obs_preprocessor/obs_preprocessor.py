from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from PIL import Image

from habitat import Config
from habitat.core.simulator import Observations

import home_robot.agent.utils.pose_utils as pu
from .constants import (
    goal_id_to_goal_name,
    goal_id_to_coco_id,
    frame_color_palette,
    MIN_DEPTH_REPLACEMENT_VALUE,
    MAX_DEPTH_REPLACEMENT_VALUE,
)


class ObsPreprocessor:
    """
    This class preprocesses observations - it can either be integrated in the
    agent or an environment.
    """

    def __init__(self, config: Config, num_environments: int, device: torch.device):
        self.num_environments = num_environments
        self.device = device
        self.num_sem_categories = config.ENVIRONMENT.num_sem_categories
        self.frame_height = config.ENVIRONMENT.frame_height
        self.frame_width = config.ENVIRONMENT.frame_width
        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth
        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS

        if not self.ground_truth_semantics:
            from home_robot.agent.perception.detection.coco_maskrcnn.coco_maskrcnn import (
                COCOMaskRCNN,
            )

            self.segmentation = COCOMaskRCNN(
                sem_pred_prob_thr=0.9,
                sem_gpu_id=(-1 if device == torch.device("cpu") else device.index),
                visualize=True,
            )

        self.one_hot_encoding = torch.eye(self.num_sem_categories, device=self.device)
        self.color_palette = [int(x * 255.0) for x in frame_color_palette]

        self.last_poses = None
        self.last_actions = None
        self.instance_id_to_category_id = None

    def reset(self):
        self.last_poses = [np.zeros(3)] * self.num_environments
        self.last_actions = [None] * self.num_environments
        self.instance_id_to_category_id = None

    def set_instance_id_to_category_id(self, instance_id_to_category_id):
        self.instance_id_to_category_id = instance_id_to_category_id.to(self.device)

    def preprocess(
        self,
        obs: List[Observations],
    ) -> Tuple[Tensor, np.ndarray, Tensor, Tensor, List[str]]:
        """
        Preprocess observations of a single timestep batched across
        environments.

        Arguments:
            obs: list of observations of length num_environments

        Returns:
            obs_preprocessed: frame containing (RGB, depth, segmentation) of
             shape (num_environments, 3 + 1 + num_sem_categories, frame_height,
             frame_width)
            semantic_frame: semantic frame visualization of shape
             (num_environments, frame_height, frame_width, 3)
            pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
             of shape (num_environments, 3)
            goal: semantic goal category of shape (num_environments,)
            goal_name: list of semantic goal strings of length num_environments
        """
        obs_preprocessed, semantic_frame = self.preprocess_frame(obs)
        pose_delta, self.last_poses = self.preprocess_pose(obs, self.last_poses)
        goal, goal_name = self.preprocess_goal(obs)
        return (obs_preprocessed, semantic_frame, pose_delta, goal, goal_name)

    def preprocess_sequence(
        self, seq_obs: List[Observations]
    ) -> Tuple[Tensor, np.ndarray, Tensor, Tensor, str]:
        """
        Preprocess observations of a single environment batched across time.

        Arguments:
            seq_obs: list of observations of length sequence_length

        Returns:
            seq_obs_preprocessed: frame containing (RGB, depth, segmentation) of
             shape (sequence_length, 3 + 1 + num_sem_categories, frame_height,
             frame_width)
            seq_semantic_frame: semantic frame visualization of shape
             (sequence_length, frame_height, frame_width, 3)
            seq_pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
             of shape (sequence_length, 3)
            goal: semantic goal category ID
            goal_name: semantic goal category
        """
        assert self.num_environments == 1
        sequence_length = len(seq_obs)
        seq_obs_preprocessed, seq_semantic_frame = self.preprocess_frame(seq_obs)

        seq_pose_delta = torch.zeros(sequence_length, 3)
        for t in range(sequence_length):
            seq_pose_delta[t], self.last_poses = self.preprocess_pose(
                [seq_obs[t]], self.last_poses
            )

        goal, goal_name = self.preprocess_goal([seq_obs[0]])
        goal_name = goal_name[0] if goal_name is not None else goal_name
        return (
            seq_obs_preprocessed,
            seq_semantic_frame,
            seq_pose_delta,
            goal,
            goal_name,
        )

    def preprocess_goal(self, obs: List[Observations]) -> Tuple[Tensor, List[str]]:
        if "objectgoal" in obs[0]:
            goal = torch.tensor([goal_id_to_coco_id[ob["objectgoal"][0]] for ob in obs])
            goal_name = [goal_id_to_goal_name[ob["objectgoal"][0]] for ob in obs]
        else:
            goal, goal_name = None, None
        return goal, goal_name

    def preprocess_frame(self, obs: List[Observations]) -> Tuple[Tensor, np.ndarray]:
        """Preprocess frame information in the observation."""

        def preprocess_depth(depth):
            # Rescale depth from [0.0, 1.0] to [min_depth, max_depth]
            rescaled_depth = (
                self.min_depth * 100.0
                + depth * (self.max_depth - self.min_depth) * 100.0
            )

            # Depth at the boundaries of [min_depth, max_depth] has been
            # thresholded and should not be considered in the point cloud
            # and semantic map - the lines below ensure it's beyond
            # vision_range and does not get considered in the semantic map
            rescaled_depth[depth == 0.0] = MIN_DEPTH_REPLACEMENT_VALUE
            rescaled_depth[depth == 1.0] = MAX_DEPTH_REPLACEMENT_VALUE

            return rescaled_depth

        def downscale(rgb, depth, semantic):
            h_downscaling = env_frame_height // self.frame_height
            w_downscaling = env_frame_width // self.frame_width
            assert h_downscaling == w_downscaling
            assert type(h_downscaling) == int
            if h_downscaling == 1:
                return rgb, depth, semantic
            else:
                rgb = F.interpolate(
                    rgb, scale_factor=1.0 / h_downscaling, mode="bilinear"
                )
                depth = F.interpolate(
                    depth, scale_factor=1.0 / h_downscaling, mode="bilinear"
                )
                semantic = F.interpolate(
                    semantic, scale_factor=1.0 / h_downscaling, mode="nearest"
                )
                return rgb, depth, semantic

        env_frame_height, env_frame_width = obs[0]["rgb"].shape[:2]

        rgb = torch.from_numpy(np.stack([ob["rgb"] for ob in obs])).to(self.device)
        depth = torch.from_numpy(np.stack([ob["depth"] for ob in obs])).to(self.device)

        depth = preprocess_depth(depth)

        if (
            self.ground_truth_semantics
            and "semantic" in obs[0]
            and self.instance_id_to_category_id is not None
        ):
            # Ground-truth semantic segmentation (useful for debugging)
            # TODO Allow multiple environments with ground-truth segmentation
            assert "semantic" in obs[0]
            semantic = torch.from_numpy(
                np.stack([ob["semantic"] for ob in obs]).squeeze(-1).astype(np.int64)
            ).to(self.device)
            semantic = self.instance_id_to_category_id[semantic]
            semantic = self.one_hot_encoding[semantic]

            # Also need depth filtering on ground-truth segmentation
            for i in range(semantic.shape[-1]):
                depth_ = depth[0, :, :, -1]
                semantic_ = semantic[0, :, :, i]
                depth_md = torch.median(depth_[semantic_ == 1])
                if depth_md != 0:
                    filter_mask = (depth_ >= depth_md + 50) | (depth_ <= depth_md - 50)
                    # pixels = int(semantic_[filter_mask].sum().item())
                    # if pixels > 0:
                    #     print(f"Filtering out {pixels} pixels")
                    semantic[0, :, :, i][filter_mask] = 0.0

            semantic_vis = self._get_semantic_frame_vis(
                rgb[0].cpu().numpy(), semantic[0].cpu().numpy()
            )
            semantic_vis = np.expand_dims(semantic_vis, 0)

        else:
            # Predicted semantic segmentation
            semantic, semantic_vis = self.segmentation.get_prediction(
                rgb.cpu().numpy(), depth.cpu().squeeze(-1).numpy()
            )
            semantic = torch.from_numpy(semantic).to(self.device)

        rgb = rgb.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)
        semantic = semantic.permute(0, 3, 1, 2)

        rgb, depth, semantic = downscale(rgb, depth, semantic)
        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=1)

        return obs_preprocessed, semantic_vis

    def _get_semantic_frame_vis(self, rgb: np.ndarray, semantics: np.ndarray):
        """Visualize first-person semantic segmentation frame."""
        width, height = semantics.shape[:2]
        vis_content = semantics
        vis_content[:, :, -1] = 1e-5
        vis_content = vis_content.argmax(-1)
        vis = Image.new("P", (height, width))
        vis.putpalette(self.color_palette)
        vis.putdata(vis_content.flatten().astype(np.uint8))
        vis = vis.convert("RGB")
        vis = np.array(vis)
        vis = np.where(vis != 255, vis, rgb)
        vis = vis[:, :, ::-1]
        return vis

    def preprocess_pose(
        self, obs: List[Observations], last_poses: List[np.ndarray]
    ) -> Tuple[Tensor, List[np.ndarray]]:
        """Preprocess sensor pose information in the observation."""
        curr_poses = []
        pose_deltas = []

        for e in range(self.num_environments):
            curr_pose = np.array(
                [obs[e]["gps"][0], -obs[e]["gps"][1], obs[e]["compass"][0]]
            )
            pose_delta = pu.get_rel_pose_change(curr_pose, last_poses[e])
            curr_poses.append(curr_pose)
            pose_deltas.append(pose_delta)

        return torch.tensor(pose_deltas), curr_poses

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

import habitat_baselines.ml.utils.pose_utils as pu
from habitat import config
from habitat.core.env import Env
from habitat.core.simulator import Observations
from habitat_baselines.ml.visualizer.constants import GoalObjectMapping


class ObsPreprocessor:
    """
    This class preprocesses observations - it can either be integrated in the
    agent or an environment.
    """

    def __init__(
        self, config: config, num_environments: int, device: torch.device
    ):
        self.num_environments = num_environments
        self.device = device
        self.config = config
        self.frame_height = (
            config.simulator.agents.main_agent.sim_sensors.head_rgb_sensor.height
        )
        self.frame_width = (
            config.simulator.agents.main_agent.sim_sensors.head_rgb_sensor.width
        )
        self.min_depth = config.ml_environment.min_depth
        self.max_depth = config.ml_environment.max_depth
        self.ground_truth_semantics = (
            config.ml_environment.ground_truth_semantics
        )
        self.min_depth_replacement_value = (
            config.ml_environment.min_depth_replacement_value
        )
        self.max_depth_replacement_value = (
            config.ml_environment.max_depth_replacement_value
        )

        self.episodes_data_path = config.dataset.data_path
        assert (
            "floorplanner" in self.episodes_data_path
            or "hm3d" in self.episodes_data_path
        )
        # if "hm3d" in self.episodes_data_path:
        #     if config.AGENT.SEMANTIC_MAP.semantic_categories == "coco_indoor":
        #         self.semantic_category_mapping = HM3DtoCOCOIndoor()
        #     # elif config.AGENT.SEMANTIC_MAP.semantic_categories == "longtail_indoor":
        #     #     self.semantic_category_mapping = HM3DtoLongTailIndoor()
        #     else:
        #         raise NotImplementedError
        # elif "floorplanner" in self.episodes_data_path:
        # if config.AGENT.SEMANTIC_MAP.semantic_categories == "mukul_indoor":
        self.semantic_category_mapping = GoalObjectMapping()
        # TODO: Hardcoded for now, read from episodes
        self._obj_name_to_id_mapping = {'action_figure': 0, 'cup': 1, 'dishtowel': 2, 'hat': 3, 'sponge': 4, 'stuffed_toy': 5, 'tape': 6, 'vase': 7}
        self._rec_name_to_id_mapping =  {'armchair': 0, 'armoire': 1, 'bar_stool': 2, 'coffee_table': 3, 'desk': 4, 'dining_table': 5, 'kitchen_island': 6, 'sofa': 7, 'stool': 8}
        self._obj_id_to_name_mapping = {k:v for  v, k in self._obj_name_to_id_mapping.items()}
        self._rec_id_to_name_mapping = {k:v for  v, k in self._rec_name_to_id_mapping.items()}

        # else:
        #     raise NotImplementedError

        # if not self.ground_truth_semantics:
        #     # from home_robot.agent.perception.detection.coco_maskrcnn.coco_maskrcnn import (
        #     #     COCOMaskRCNN,
        #     # )
        #     # self.segmentation = COCOMaskRCNN(
        #     #     sem_pred_prob_thr=0.9,
        #     #     sem_gpu_id=(-1 if device == torch.device("cpu") else device.index),
        #     #     visualize=True,
        #     # )
        #     from home_robot.agent.perception.detection.detic.detic_loader import get_detic
        #     if config.AGENT.SEMANTIC_MAP.semantic_categories == "coco_indoor":
        #         self.segmentation = get_detic(
        #             vocabulary="coco",
        #             sem_gpu_id=(-1 if device == torch.device("cpu") else device.index),
        #         )
        #     elif config.AGENT.SEMANTIC_MAP.semantic_categories == "longtail_indoor":
        #         # self.segmentation = get_detic(
        #         #     vocabulary="custom",
        #         #     custom_vocabulary=",".join(long_tail_indoor_categories),
        #         #     sem_gpu_id=(-1 if device == torch.device("cpu") else device.index),
        #         # )
        #         raise NotImplementedError
        #     elif config.AGENT.SEMANTIC_MAP.semantic_categories == "mukul_indoor":
        #         self.segmentation = get_detic(
        #             vocabulary="custom",
        #             custom_vocabulary=",".join(mukul_33categories_padded),
        #             sem_gpu_id=(-1 if device == torch.device("cpu") else device.index),
        #         )

        self.one_hot_encoding = torch.eye(
            self.semantic_category_mapping.num_sem_categories,
            device=self.device,
        )

        self.last_poses = None
        self.last_actions = None

    def reset(self, env: Env):
        self.last_poses = [np.zeros(3)] * self.num_environments
        self.last_actions = [None] * self.num_environments
        self.semantic_category_mapping.reset_instance_id_to_category_id(env)

    def preprocess(
        self,
        obs: List[Observations]
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
        (
            obs_preprocessed,
            semantic_frame,
            third_person_rgb_frame,
        ) = self.preprocess_frame(obs)
        pose_delta, self.last_poses = self.preprocess_pose(
            obs, self.last_poses
        )

        object_goal, recep_goal,  goal_name = self.preprocess_goal(obs, self.config.task.goal_type)

        return (
            obs_preprocessed,
            semantic_frame,
            third_person_rgb_frame,
            pose_delta,
            object_goal,
            recep_goal,
            goal_name,
        )

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
        (
            seq_obs_preprocessed,
            seq_semantic_frame,
            third_person_rgb_frame,
        ) = self.preprocess_frame(seq_obs)

        seq_pose_delta = torch.zeros(sequence_length, 3)
        for t in range(sequence_length):
            seq_pose_delta[t], self.last_poses = self.preprocess_pose(
                [seq_obs[t]], self.last_poses
            )

        object_goal, recep_goal, goal_name = self.preprocess_goal([seq_obs[0]])
        goal_name = goal_name[0] if goal_name is not None else goal_name
        return (
            seq_obs_preprocessed,
            seq_semantic_frame,
            seq_pose_delta,
            object_goal,
            recep_goal,
            goal_name,
        )

    def preprocess_goal(
        self, obs: List[Observations],
        goal_type
    ) -> Tuple[Tensor, List[str]]:
        assert "object_category" in obs[0]
        object_goal_ids, rec_goal_ids, goal_names = [], [], []

        for ob in obs:
            if goal_type in ['object', 'object_on_recep']:
                goal_name = self._obj_id_to_name_mapping[ob["object_category"][0]]
                obj_goal_id = 1  # semantic sensor returns binary mask for goal object
                object_goal_ids.append(obj_goal_id)
            if goal_type ==  'object_on_recep':
                goal_name =  self._obj_id_to_name_mapping[ob["object_category"][0]] + " on " +  self._rec_id_to_name_mapping[ob["start_receptacle"][0]]
                rec_goal_id = 2
                rec_goal_ids.append(rec_goal_id)
            if goal_type == 'recep':
                goal_name = self._rec_id_to_name_mapping[ob['goal_receptacle'][0]]
                rec_goal_id = 3
                object_goal_ids = None
                rec_goal_ids.append(rec_goal_id)
            goal_names.append(goal_name)
        if goal_type  == 'object':
            rec_goal_ids = None
        if object_goal_ids is not None:
            object_goal_ids = torch.tensor(object_goal_ids)
        if rec_goal_ids is not None:
            rec_goal_ids = torch.tensor(rec_goal_ids)
        return object_goal_ids, rec_goal_ids, goal_names

    def preprocess_frame(
        self, obs: List[Observations]
    ) -> Tuple[Tensor, np.ndarray]:
        """Preprocess frame information in the observation."""

        def preprocess_depth(depth):
            # Attempt to deal with black holes in depth: set the holes with zero depth
            # to the max depth value in the image row
            # zero_mask = depth == 0.
            # row_max = depth.max(axis=1, keepdims=True).values
            # depth += zero_mask * row_max
            # zero_mask = depth == 0.
            # col_max = depth.max(axis=2, keepdims=True).values
            # depth += zero_mask * col_max
            # Rescale depth from [0.0, 1.0] to [min_depth, max_depth]
            rescaled_depth = (
                self.min_depth * 100.0
                + depth * (self.max_depth - self.min_depth) * 100.0
            )

            # Depth at the boundaries of [min_depth, max_depth] has been
            # thresholded and should not be considered in the point cloud
            # and semantic map - the lines below ensure it's beyond
            # vision_range and does not get considered in the semantic map
            rescaled_depth[depth == 0.0] = self.min_depth_replacement_value
            rescaled_depth[depth == 1.0] = self.max_depth_replacement_value

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

        env_frame_height, env_frame_width = obs[0]["robot_head_rgb"].shape[:2]
        rgb = torch.from_numpy(
            np.stack([ob["robot_head_rgb"] for ob in obs])
        ).to(self.device)
        depth = torch.from_numpy(
            np.stack([ob["robot_head_depth"] for ob in obs])
        ).to(self.device)
        third_person_rgb_frame = np.stack(
            [ob["robot_third_rgb"] for ob in obs]
        )
        depth = preprocess_depth(depth)

        if (
            self.ground_truth_semantics
            and "object_segmentation" in obs[0]
            and "goal_recep_segmentation" in obs[0]
            and "start_recep_segmentation" in obs[0]
            and self.semantic_category_mapping.instance_id_to_category_id
            is not None
        ):
            # Ground-truth semantic segmentation (useful for debugging)
            # TODO Allow multiple environments with ground-truth segmentation
            semantic = torch.from_numpy(
                np.stack([ob["object_segmentation"] for ob in obs])
                .squeeze(-1)
                .astype(np.int64)
            ).to(self.device)
            start_recep_seg = torch.from_numpy(
                np.stack([ob["start_recep_segmentation"] for ob in obs])
                .squeeze(-1)
                .astype(np.int64)
            ).to(self.device)

            goal_recep_seg = torch.from_numpy(
                np.stack([ob["goal_recep_segmentation"] for ob in obs])
                .squeeze(-1)
                .astype(np.int64)
            ).to(self.device)
            instance_id_to_category_id = (
                self.semantic_category_mapping.instance_id_to_category_id.to(
                    self.device
                )
            )
            # Assign semantic id of 1 for object_category, 2 for start_receptacle, 3 for goal_receptacle
            semantic  = semantic  + start_recep_seg * 2 + goal_recep_seg * 3
            semantic = instance_id_to_category_id[semantic]
            semantic = self.one_hot_encoding[semantic]

            # Also need depth filtering on ground-truth segmentation
            # for i in range(semantic.shape[-1]):
            #     depth_ = depth[0, :, :, -1]
            #     semantic_ = semantic[0, :, :, i]
            #     depth_md = torch.median(depth_[semantic_ == 1])
            #     if depth_md != 0:
            #         filter_mask = (depth_ >= depth_md + 50) | (depth_ <= depth_md - 50)
            #         semantic[0, :, :, i][filter_mask] = 0.0

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
        return obs_preprocessed, semantic_vis, third_person_rgb_frame

    def _get_semantic_frame_vis(self, rgb: np.ndarray, semantics: np.ndarray):
        """Visualize first-person semantic segmentation frame."""
        width, height = semantics.shape[:2]
        vis_content = semantics
        vis_content[:, :, -1] += 1e-5  # Assumes the last category is "other"
        vis_content = vis_content.argmax(-1)
        vis = Image.new("P", (height, width))
        vis.putpalette(self.semantic_category_mapping.frame_color_palette)
        vis.putdata(vis_content.flatten().astype(np.uint8))
        mask = np.array(vis)
        mask = (
            mask == self.semantic_category_mapping.num_sem_categories - 1
        ).astype(np.uint8) * 255
        mask = Image.fromarray(mask)
        rgb_pil = Image.fromarray(rgb)
        vis = vis.convert("RGB")
        vis.paste(rgb_pil, mask=mask)
        vis = np.array(vis)
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
                [
                    obs[e]["robot_start_gps"][
                        1
                    ],  # because camera and front of robot are orthogonal
                    obs[e]["robot_start_gps"][0],
                    obs[e]["robot_start_compass"][0],
                ]
            )
            pose_delta = pu.get_rel_pose_change(curr_pose, last_poses[e])
            curr_poses.append(curr_pose)
            pose_deltas.append(pose_delta)
            # print("previous pose:", last_poses[e])
            # print("current pose:", curr_pose)
            # print("pose_delta:", pose_delta)
            # print("--------")

        return torch.tensor(pose_deltas), curr_poses

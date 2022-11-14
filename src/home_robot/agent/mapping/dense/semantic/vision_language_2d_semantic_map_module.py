from typing import Tuple
import torch
from torch import Tensor, IntTensor
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import skimage.morphology

import agent.mapping.metric.depth_utils as du
import agent.mapping.metric.rotation_utils as ru
import agent.mapping.metric.map_utils as mu
import agent.utils.pose_utils as pu
from agent.perception.segmentation.lseg import load_lseg_for_inference


class VisionLanguage2DSemanticMapModule(nn.Module):
    """
    This class is responsible for updating a dense 2D semantic map with
    vision-language features, the local and global maps and poses, and generating
    map features — it is a stateless PyTorch module with no trainable parameters.

    Map proposed in:
    Visual Language Maps for Robot Navigation
    https://arxiv.org/pdf/2210.05714.pdf
    https://github.com/vlmaps/vlmaps
    """

    def __init__(
        self,
        lseg_checkpoint_path: str,
        lseg_features_dim: int,
        frame_height: int,
        frame_width: int,
        camera_height: int,
        hfov: int,
        map_size_cm: int,
        map_resolution: int,
        vision_range: int,
        global_downscaling: int,
        du_scale: int,
        exp_pred_threshold: float,
        map_pred_threshold: float,
    ):
        """
        Arguments:
            lseg_checkpoint_path: pre-trained LSeg checkpoint path
            lseg_features_dim: number of dimensions of vision-language
             (CLIP) features
            frame_height: first-person frame height
            frame_width: first-person frame width
            camera_height: camera sensor height (in metres)
            hfov: horizontal field of view (in degrees)
            map_size_cm: global map size (in centimetres)
            map_resolution: size of map bins (in centimeters)
            vision_range: diameter of the circular region of the local map
             that is visible by the agent located in its center (unit is
             the number of local map cells)
            global_downscaling: ratio of global over local map
            du_scale: frame downscaling before projecting to point cloud
            exp_pred_threshold: number of depth points to be in bin to
             consider it as explored
            map_pred_threshold: number of depth points to be in bin to
             consider it as obstacle
        """
        super().__init__()

        self.screen_h = frame_height
        self.screen_w = frame_width
        self.camera_matrix = du.get_camera_matrix(self.screen_w, self.screen_h, hfov)

        self.lseg = load_lseg_for_inference(
            lseg_checkpoint_path, torch.device("cpu"), visualize=False
        )
        self.lseg_features_dim = lseg_features_dim

        self.map_size_parameters = mu.MapSizeParameters(
            map_resolution, map_size_cm, global_downscaling
        )
        self.resolution = map_resolution
        self.global_map_size_cm = map_size_cm
        self.global_downscaling = global_downscaling
        self.local_map_size_cm = self.global_map_size_cm // self.global_downscaling
        self.global_map_size = self.global_map_size_cm // self.resolution
        self.local_map_size = self.local_map_size_cm // self.resolution
        self.xy_resolution = self.z_resolution = map_resolution
        self.vision_range = vision_range
        self.du_scale = du_scale
        self.exp_pred_threshold = exp_pred_threshold
        self.map_pred_threshold = map_pred_threshold

        self.agent_height = camera_height * 100.0
        self.max_voxel_height = int(360 / self.z_resolution)
        self.min_voxel_height = int(-40 / self.z_resolution)
        self.min_mapped_height = int(25 / self.z_resolution - self.min_voxel_height)
        self.max_mapped_height = int(
            (self.agent_height + 1) / self.z_resolution - self.min_voxel_height
        )
        self.shift_loc = [self.vision_range * self.xy_resolution // 2, 0, np.pi / 2.0]

    @torch.no_grad()
    def forward(
        self,
        seq_obs: Tensor,
        seq_pose_delta: Tensor,
        seq_dones: Tensor,
        seq_update_global: Tensor,
        init_local_map: Tensor,
        init_global_map: Tensor,
        init_local_pose: Tensor,
        init_global_pose: Tensor,
        init_lmb: Tensor,
        init_origins: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, IntTensor, Tensor]:
        """Update maps and poses with a sequence of observations and generate map
        features at each time step.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth) of shape
             (batch_size, sequence_length, 3 + 1, frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (batch_size, sequence_length, 3)
            seq_dones: sequence of (batch_size, sequence_length) binary flags
             that indicate episode restarts
            seq_update_global: sequence of (batch_size, sequence_length) binary
             flags that indicate whether to update the global map and pose
            init_local_map: initial local map before any updates of shape
             (batch_size, 5 + lseg_features_dim, M, M)
            init_global_map: initial global map before any updates of shape
             (batch_size, 5 + lseg_features_dim, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (batch_size, 3)
            init_global_pose: initial global pose before any updates of shape
             (batch_size, 3)
            init_lmb: initial local map boundaries of shape (batch_size, 4)
            init_origins: initial local map origins of shape (batch_size, 3)

        Returns:
            seq_map_features: sequence of semantic map features of shape
             (batch_size, sequence_length, 8 + lseg_features_dim, M, M)
            final_local_map: final local map after all updates of shape
             (batch_size, 5 + lseg_features_dim, M, M)
            final_global_map: final global map after all updates of shape
             (batch_size, 5 + lseg_features_dim, M * ds, M * ds)
            seq_local_pose: sequence of local poses of shape
             (batch_size, sequence_length, 3)
            seq_global_pose: sequence of global poses of shape
             (batch_size, sequence_length, 3)
            seq_lmb: sequence of local map boundaries of shape
             (batch_size, sequence_length, 4)
            seq_origins: sequence of local map origins of shape
             (batch_size, sequence_length, 3)
        """
        batch_size, sequence_length = seq_obs.shape[:2]
        device, dtype = seq_obs.device, seq_obs.dtype

        map_features_channels = 8 + self.lseg_features_dim
        seq_map_features = torch.zeros(
            batch_size,
            sequence_length,
            map_features_channels,
            self.local_map_size,
            self.local_map_size,
            device=device,
            dtype=dtype,
        )
        seq_local_pose = torch.zeros(batch_size, sequence_length, 3, device=device)
        seq_global_pose = torch.zeros(batch_size, sequence_length, 3, device=device)
        seq_lmb = torch.zeros(
            batch_size, sequence_length, 4, device=device, dtype=torch.int32
        )
        seq_origins = torch.zeros(batch_size, sequence_length, 3, device=device)

        local_map, local_pose = init_local_map.clone(), init_local_pose.clone()
        global_map, global_pose = init_global_map.clone(), init_global_pose.clone()
        lmb, origins = init_lmb.clone(), init_origins.clone()

        for t in range(sequence_length):
            # Reset map and pose for episodes done at time step t
            for e in range(batch_size):
                if seq_dones[e, t]:
                    mu.init_map_and_pose_for_env(
                        e,
                        local_map,
                        global_map,
                        local_pose,
                        global_pose,
                        lmb,
                        origins,
                        self.map_size_parameters,
                    )

            local_map, local_pose = self._update_local_map_and_pose(
                seq_obs[:, t], seq_pose_delta[:, t], local_map, local_pose
            )

            for e in range(batch_size):
                if seq_update_global[e, t]:
                    self._update_global_map_and_pose_for_env(
                        e, local_map, global_map, local_pose, global_pose, lmb, origins
                    )

            seq_local_pose[:, t] = local_pose
            seq_global_pose[:, t] = global_pose
            seq_lmb[:, t] = lmb
            seq_origins[:, t] = origins
            seq_map_features[:, t] = self._get_map_features(local_map, global_map)

        return (
            seq_map_features,
            local_map,
            global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        )

    def _update_local_map_and_pose(
        self, obs: Tensor, pose_delta: Tensor, prev_map: Tensor, prev_pose: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Update local map and sensor pose given a new observation using parameter-free
        differentiable projective geometry.

        Args:
            obs: current frame containing (rgb, depth) of shape
             (batch_size, 3 + 1, frame_height, frame_width)
            pose_delta: delta in pose since last frame of shape (batch_size, 3)
            prev_map: previous local map of shape
             (batch_size, 5 + lseg_features_dim, M, M)
            prev_pose: previous pose of shape (batch_size, 3)

        Returns:
            current_map: current local map updated with current observation
             and location of shape (batch_size, 5 + lseg_features_dim, M, M)
            current_pose: current pose updated with pose delta of shape (batch_size, 3)
        """
        batch_size, _, h, w = obs.size()
        device, dtype = obs.device, obs.dtype

        rgb = obs[:, :3, :, :].float()
        depth = obs[:, 3, :, :].float()

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, device, scale=self.du_scale
        )

        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, 0, device
        )

        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, device
        )

        voxel_channels = 1 + self.lseg_features_dim

        init_grid = torch.zeros(
            batch_size,
            voxel_channels,
            self.vision_range,
            self.vision_range,
            self.max_voxel_height - self.min_voxel_height,
            device=device,
            dtype=torch.float32,
        )
        feat = torch.ones(
            batch_size,
            voxel_channels,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale,
            device=device,
            dtype=torch.float32,
        )

        # TODO Batch LSeg inference across time
        pixel_features = self.lseg.encode(rgb.permute((0, 2, 3, 1)))

        feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(pixel_features).view(
            batch_size, self.lseg_features_dim, h // self.du_scale * w // self.du_scale
        )

        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = XYZ_cm_std[..., :2] / self.xy_resolution
        XYZ_cm_std[..., :2] = (
            (XYZ_cm_std[..., :2] - self.vision_range // 2.0) / self.vision_range * 2.0
        )
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / self.z_resolution
        XYZ_cm_std[..., 2] = (
            (
                XYZ_cm_std[..., 2]
                - (self.max_voxel_height + self.min_voxel_height) // 2.0
            )
            / (self.max_voxel_height - self.min_voxel_height)
            * 2.0
        )
        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(
            XYZ_cm_std.shape[0],
            XYZ_cm_std.shape[1],
            XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3],
        )

        voxels = du.splat_feat_nd(init_grid, feat, XYZ_cm_std).transpose(2, 3)

        agent_height_proj = voxels[
            ..., self.min_mapped_height : self.max_mapped_height
        ].sum(4)
        all_height_proj = voxels.sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold

        agent_view = torch.zeros(
            batch_size,
            5 + self.lseg_features_dim,
            self.local_map_size_cm // self.xy_resolution,
            self.local_map_size_cm // self.xy_resolution,
            device=device,
            dtype=dtype,
        )

        x1 = self.local_map_size_cm // (self.xy_resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.local_map_size_cm // (self.xy_resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 5 : 5 + self.lseg_features_dim, y1:y2, x1:x2] = (
            all_height_proj[:, 1 : 1 + self.lseg_features_dim]
        )

        current_pose = pu.get_new_pose_batch(prev_pose.clone(), pose_delta)
        st_pose = current_pose.clone().detach()

        st_pose[:, :2] = -(
            (
                st_pose[:, :2] * 100.0 / self.xy_resolution
                - self.local_map_size_cm // (self.xy_resolution * 2)
            )
            / (self.local_map_size_cm // (self.xy_resolution * 2))
        )
        st_pose[:, 2] = 90.0 - (st_pose[:, 2])

        rot_mat, trans_mat = ru.get_grid(st_pose, agent_view.size(), dtype)
        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        translated[:, :4] = torch.clamp(translated[:, :4], min=0.0, max=1.0)

        # Aggregation:
        #  0-3: max for obstacle, explored, past locations
        #  4: +1 count for updated cells
        #  5-517: mean for CLIP map cell features
        current_map = prev_map.clone()
        current_map[:, :4], _ = torch.max(
            torch.cat(
                (
                    prev_map[:, :4].unsqueeze(1),
                    translated[:, :4].unsqueeze(1)
                ),
                1
            ),
            1
        )
        for e in range(batch_size):
            update_mask = translated[e, 5:].sum(0) > 0

            # Average features of all previous views and most recent view
            current_map[e, 5:, update_mask] = (
                (prev_map[e, 5:, update_mask] * prev_map[e, 4, update_mask] +
                 translated[e, 5:, update_mask]) /
                (prev_map[e, 4, update_mask] + 1)
            )

            # Keep most recent view only
            # current_map[e, 5:, update_mask] = translated[e, 5:, update_mask]

            current_map[e, 4, update_mask] += 1

        # Reset current location
        current_map[:, 2, :, :].fill_(0.0)
        curr_loc = current_pose[:, :2]
        curr_loc = (curr_loc * 100.0 / self.xy_resolution).int()
        for e in range(batch_size):
            x, y = curr_loc[e]
            current_map[e, 2:4, y - 2 : y + 3, x - 2 : x + 3].fill_(1.0)

            # Set a disk around the agent to explored
            try:
                radius = 10
                explored_disk = torch.from_numpy(skimage.morphology.disk(radius))
                current_map[
                    e, 1, y - radius : y + radius + 1, x - radius : x + radius + 1
                ][explored_disk == 1] = 1
            except IndexError:
                pass

        return current_map, current_pose

    def _update_global_map_and_pose_for_env(
        self,
        e: int,
        local_map: Tensor,
        global_map: Tensor,
        local_pose: Tensor,
        global_pose: Tensor,
        lmb: Tensor,
        origins: Tensor,
    ):
        """Update global map and pose and re-center local map and pose for a
        particular environment.
        """
        global_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]] = local_map[e]
        global_pose[e] = local_pose[e] + origins[e]
        mu.recenter_local_map_and_pose_for_env(
            e,
            local_map,
            global_map,
            local_pose,
            global_pose,
            lmb,
            origins,
            self.map_size_parameters,
        )

    def _get_map_features(self, local_map: Tensor, global_map: Tensor) -> Tensor:
        """Get global and local map features.

        Arguments:
            local_map: local map of shape
             (batch_size, 5 + lseg_features_dim, M, M)
            global_map: global map of shape
             (batch_size, 5 + lseg_features_dim, M * ds, M * ds)

        Returns:
            map_features: semantic map features of shape
             (batch_size, 8 + lseg_features_dim, M, M)
        """
        map_features_channels = 8 + self.lseg_features_dim

        map_features = torch.zeros(
            local_map.size(0),
            map_features_channels,
            self.local_map_size,
            self.local_map_size,
            device=local_map.device,
            dtype=local_map.dtype,
        )

        # Local obstacles, explored area, and current and past position
        map_features[:, 0:4, :, :] = local_map[:, 0:4, :, :]
        # Global obstacles, explored area, and current and past position
        map_features[:, 4:8, :, :] = nn.MaxPool2d(self.global_downscaling)(
            global_map[:, 0:4, :, :]
        )
        # Local CLIP map cell features
        map_features[:, 8:, :, :] = local_map[:, 5:, :, :]

        return map_features.detach()

import time

import torch.nn as nn

from habitat_baselines.ml.mapping.dense.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)
from habitat_baselines.ml.navigation_policy.object_navigation.objectnav_frontier_exploration_policy import (
    ObjectNavFrontierExplorationPolicy,
)


class ObjectNavAgentModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        rgb_sensor = (
            config.habitat.simulator.agents.main_agent.sim_sensors.head_rgb_sensor
        )
        self.semantic_map_module = Categorical2DSemanticMapModule(
            frame_height=rgb_sensor.height,
            frame_width=rgb_sensor.width,
            camera_height=rgb_sensor.position[1],
            hfov=rgb_sensor.hfov,
            num_sem_categories=config.habitat.ml_environment.num_sem_categories,
            map_size_cm=config.habitat.ml_environment.map_size_cm,
            map_resolution=config.habitat.ml_environment.map_resolution,
            vision_range=config.habitat.ml_environment.vision_range,
            global_downscaling=config.habitat.ml_environment.global_downscaling,
            du_scale=config.habitat.ml_environment.du_scale,
            cat_pred_threshold=config.habitat.ml_environment.cat_pred_threshold,
            exp_pred_threshold=config.habitat.ml_environment.exp_pred_threshold,
            map_pred_threshold=config.habitat.ml_environment.map_pred_threshold,
        )
        self.policy = ObjectNavFrontierExplorationPolicy()

    @property
    def goal_update_steps(self):
        return self.policy.goal_update_steps

    def forward(
        self,
        seq_obs,
        seq_pose_delta,
        seq_dones,
        seq_update_global,
        init_local_map,
        init_global_map,
        init_local_pose,
        init_global_pose,
        init_lmb,
        init_origins,
        seq_object_goal_category=None,
        seq_recep_goal_category=None
    ):
        """Update maps and poses with a sequence of observations, and predict
        high-level goals from map features.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, segmentation)
             of shape (batch_size, sequence_length, 3 + 1 + num_sem_categories,
             frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (batch_size, sequence_length, 3)
            seq_goal_category: sequence of goal categories of shape
             (batch_size, sequence_length, 1)
            seq_dones: sequence of (batch_size, sequence_length) done flags that
             indicate episode restarts
            seq_update_global: sequence of (batch_size, sequence_length) binary
             flags that indicate whether to update the global map and pose
            init_local_map: initial local map before any updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            init_global_map: initial global map before any updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (batch_size, 3)
            init_global_pose: initial global pose before any updates of shape
             (batch_size, 3)
            init_lmb: initial local map boundaries of shape (batch_size, 4)
            init_origins: initial local map origins of shape (batch_size, 3)

        Returns:
            seq_goal_map: sequence of binary maps encoding goal(s) of shape
             (batch_size, sequence_length, M, M)
            seq_found_goal: binary variables to denote whether we found the object
             goal category of shape (batch_size, sequence_length)
            final_local_map: final local map after all updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            final_global_map: final global map after all updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            seq_local_pose: sequence of local poses of shape
             (batch_size, sequence_length, 3)
            seq_global_pose: sequence of global poses of shape
             (batch_size, sequence_length, 3)
            seq_lmb: sequence of local map boundaries of shape
             (batch_size, sequence_length, 4)
            seq_origins: sequence of local map origins of shape
             (batch_size, sequence_length, 3)
        """
        # t0 = time.time()

        # Update map with observations and generate map features
        batch_size, sequence_length = seq_obs.shape[:2]
        (
            seq_map_features,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.semantic_map_module(
            seq_obs,
            seq_pose_delta,
            seq_dones,
            seq_update_global,
            init_local_map,
            init_global_map,
            init_local_pose,
            init_global_pose,
            init_lmb,
            init_origins,
        )

        # t1 = time.time()
        # print(f"[Semantic mapping] Total time: {t1 - t0:.2f}")

        # Predict high-level goals from map features
        # batched across sequence length x num environments
        map_features = seq_map_features.flatten(0, 1)
        if seq_object_goal_category is not None:
            seq_object_goal_category = seq_object_goal_category.flatten(0, 1)
        if seq_recep_goal_category is not None:
            seq_recep_goal_category = seq_recep_goal_category.flatten(0, 1)
        goal_map, found_goal = self.policy(map_features, seq_object_goal_category, seq_recep_goal_category)
        seq_goal_map = goal_map.view(
            batch_size, sequence_length, *goal_map.shape[-2:]
        )
        seq_found_goal = found_goal.view(batch_size, sequence_length)

        # t2 = time.time()
        # print(f"[Policy] Total time: {t2 - t1:.2f}")

        return (
            seq_goal_map,
            seq_found_goal,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        )

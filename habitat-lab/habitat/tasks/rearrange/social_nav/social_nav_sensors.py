#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.articulated_agents.humanoids.kinematic_humanoid import (
    KinematicHumanoid,
)
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.multi_agent_sensors import DidAgentsCollide
from habitat.tasks.rearrange.rearrange_sensors import RearrangeReward
from habitat.tasks.rearrange.social_nav.utils import (
    robot_human_vec_dot_product,
)
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import (
    DistToGoal,
    NavToPosSucc,
    RotDistToGoal,
)
from habitat.tasks.rearrange.utils import (
    UsesArticulatedAgentInterface,
    batch_transform_point,
)
from habitat.tasks.utils import cartesian_to_polar

BASE_ACTION_NAME = "base_velocity"


@registry.register_measure
class SocialNavReward(RearrangeReward):
    """
    Reward that gives a continuous reward for the social navigation task.
    """

    cls_uuid: str = "social_nav_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return SocialNavReward.cls_uuid

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = kwargs["config"]
        # Get the config and setup the hyperparameters
        self._config = config
        self._sim = kwargs["sim"]
        self._safe_dis_min = config.safe_dis_min
        self._safe_dis_max = config.safe_dis_max
        self._safe_dis_reward = config.safe_dis_reward
        self._facing_human_dis = config.facing_human_dis
        self._facing_human_reward = config.facing_human_reward
        self._toward_human_reward = config.toward_human_reward
        self._near_human_bonus = config.near_human_bonus
        self._explore_reward = config.explore_reward
        self._use_geo_distance = config.use_geo_distance
        self._collide_penalty = config.collide_penalty
        # Record the previous distance to human
        self._prev_dist = -1.0
        self._robot_idx = config.robot_idx
        self._human_idx = config.human_idx
        # Add exploration reward dictionary tracker
        self._visited_pos = set()

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._prev_dist = -1.0
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )
        # Reset the location visit tracker for the agent
        self._visited_pos = set()

    def update_metric(self, *args, episode, task, observations, **kwargs):
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

        # Get the pos
        use_k_human = f"agent_{self._human_idx}_localization_sensor"
        human_pos = observations[use_k_human][:3]
        use_k_robot = f"agent_{self._robot_idx}_localization_sensor"
        robot_pos = observations[use_k_robot][:3]

        # If we consider using geo distance
        if self._use_geo_distance:
            path = habitat_sim.ShortestPath()
            path.requested_start = np.array(robot_pos)
            path.requested_end = human_pos
            found_path = self._sim.pathfinder.find_path(path)

        # Compute the distance between the robot and the human
        if self._use_geo_distance and found_path:
            dis = self._sim.geodesic_distance(robot_pos, human_pos)
        else:
            dis = np.linalg.norm(human_pos - robot_pos)

        # Start social nav reward
        social_nav_reward = 0.0

        # Componet 1: Social nav reward three stage design
        if dis >= self._safe_dis_min and dis < self._safe_dis_max:
            # If the distance is within the safety interval
            social_nav_reward += self._safe_dis_reward
        elif dis < self._safe_dis_min:
            # If the distance is too samll
            social_nav_reward += dis - self._prev_dist
        else:
            # if the distance is too large
            social_nav_reward += self._prev_dist - dis
        social_nav_reward = (
            self._config.toward_human_reward * social_nav_reward
        )

        # Componet 2: Social nav reward for facing human
        if dis < self._facing_human_dis and self._facing_human_reward != -1:
            base_T = self._sim.get_agent_data(
                self.agent_id
            ).articulated_agent.base_transformation
            # Dot product
            social_nav_reward += (
                self._facing_human_reward
                * robot_human_vec_dot_product(robot_pos, human_pos, base_T)
            )

        # Componet 3: Social nav reward bonus for getting closer to human
        if (
            dis < self._facing_human_dis
            and self._facing_human_reward != -1
            and self._near_human_bonus != -1
        ):
            social_nav_reward += self._near_human_bonus

        # Componet 4: Social nav reward for exploration
        # There is no exploration reward once the agent finds the human
        # round off float to nearest 0.5 in python
        robot_pos_key = (
            round(robot_pos[0] * 2) / 2,
            round(robot_pos[2] * 2) / 2,
        )
        social_nav_stats = task.measurements.measures[
            SocialNavStats.cls_uuid
        ].get_metric()
        if (
            self._explore_reward != -1
            and robot_pos_key not in self._visited_pos
            and social_nav_stats is not None
            and not social_nav_stats["has_found_human"]
        ):
            self._visited_pos.add(robot_pos_key)
            # Give the reward if the agent visits the new location
            social_nav_reward += self._explore_reward

        if self._prev_dist < 0:
            social_nav_reward = 0.0

        # Componet 5: Collision detection for two agents
        did_collide = task.measurements.measures[
            DidAgentsCollide._get_uuid()
        ].get_metric()
        if did_collide:
            task.should_end = True
            social_nav_reward -= self._collide_penalty

        self._metric += social_nav_reward

        # Update the distance
        self._prev_dist = dis  # type: ignore


@registry.register_measure
class SocialNavStats(UsesArticulatedAgentInterface, Measure):
    """
    The measure for social navigation
    """

    cls_uuid: str = "social_nav_stats"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config

        # Hyper-parameters
        self._check_human_in_frame = self._config.check_human_in_frame
        self._min_dis_human = self._config.min_dis_human
        self._max_dis_human = self._config.max_dis_human
        self._human_id = self._config.human_id
        self._human_detect_threshold = (
            self._config.human_detect_pixel_threshold
        )
        self._total_step = self._config.total_steps
        self._dis_threshold_for_backup_yield = (
            self._config.dis_threshold_for_backup_yield
        )
        self._min_abs_vel_for_yield = self._config.min_abs_vel_for_yield
        self._robot_face_human_threshold = (
            self._config.robot_face_human_threshold
        )
        self._enable_shortest_path_computation = (
            self._config.enable_shortest_path_computation
        )
        self._robot_idx = config.robot_idx
        self._human_idx = config.human_idx

        # For the variable tracking
        self._val_dict = {
            "min_start_end_episode_step": float("inf"),
            "has_found_human": False,
            "has_found_human_step": self._total_step,
            "found_human_times": 0,
            "after_found_human_times": 0,
            "step": 0,
            "step_after_found": 1,
            "dis": 0,
            "dis_after_found": 0,
            "backup_count": 0,
            "yield_count": 0,
        }

        # Robot's info
        self._prev_robot_base_T = None
        self._robot_init_pos = None
        self._robot_init_trans = None

        # Store pos of human and robot
        self.human_pos_list = []
        self.robot_pos_list = []

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return SocialNavStats.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        robot_pos = np.array(
            self._sim.get_agent_data(self.agent_id).articulated_agent.base_pos
        )

        # For the variable track
        self._val_dict = {
            "min_start_end_episode_step": float("inf"),
            "has_found_human": False,
            "has_found_human_step": 1500,
            "found_human_times": 0,
            "after_found_human_times": 0,
            "step": 0,
            "step_after_found": 1,
            "dis": 0,
            "dis_after_found": 0,
            "backup_count": 0,
            "yield_count": 0,
        }

        # Robot's info
        self._robot_init_pos = robot_pos
        self._robot_init_trans = mn.Matrix4(
            self._sim.get_agent_data(
                0
            ).articulated_agent.sim_obj.transformation
        )

        self._prev_robot_base_T = mn.Matrix4(
            self._sim.get_agent_data(
                0
            ).articulated_agent.sim_obj.transformation
        )

        # Store pos of human and robot
        self.human_pos_list = []
        self.robot_pos_list = []

        # Update metrics
        self.update_metric(*args, task=task, **kwargs)

    def _check_human_dis(self, robot_pos, human_pos):
        # We use geo geodesic distance here
        dis = self._sim.geodesic_distance(robot_pos, human_pos)
        return dis >= self._min_dis_human and dis <= self._max_dis_human

    def _check_human_frame(self, obs):
        if not self._check_human_in_frame:
            return True
        use_k = f"agent_{self._robot_idx}_articulated_agent_arm_panoptic"
        panoptic = obs[use_k]
        return (
            np.sum(panoptic == self._human_id) > self._human_detect_threshold
        )

    def _check_robot_facing_human(self, human_pos, robot_pos):
        base_T = self._sim.get_agent_data(
            self._robot_idx
        ).articulated_agent.sim_obj.transformation
        facing = (
            robot_human_vec_dot_product(robot_pos, human_pos, base_T)
            > self._robot_face_human_threshold
        )
        return facing

    def update_metric(self, *args, episode, task, observations, **kwargs):
        # Get the agent locations
        robot_pos = np.array(
            self._sim.get_agent_data(
                self._robot_idx
            ).articulated_agent.base_pos
        )
        human_pos = np.array(
            self._sim.get_agent_data(
                self._human_idx
            ).articulated_agent.base_pos
        )

        # Store the human/robot position info
        self.human_pos_list.append(human_pos)
        self.robot_pos_list.append(robot_pos)

        # Compute the distance based on the L2 norm
        dis = np.linalg.norm(robot_pos - human_pos, ord=2, axis=-1)

        # Add the current distance to compute average distance
        self._val_dict["dis"] += dis

        # Compute the robot moving velocity for backup and yiled metrics
        robot_move_vec = np.array(
            self._prev_robot_base_T.inverted().transform_point(robot_pos)
        )[[0, 2]]
        robot_move_vel = (
            np.linalg.norm(robot_move_vec)
            / (1.0 / 120.0)
            * np.sign(robot_move_vec[0])
        )

        # Compute the metrics for backing up and yield
        if (
            dis <= self._dis_threshold_for_backup_yield
            and robot_move_vel < 0.0
        ):
            self._val_dict["backup_count"] += 1
        elif (
            dis <= self._dis_threshold_for_backup_yield
            and abs(robot_move_vel) < self._min_abs_vel_for_yield
        ):
            self._val_dict["yield_count"] += 1

        # Increase the step counter
        self._val_dict["step"] += 1

        # Check if human has been found
        found_human = False
        if self._check_human_dis(
            robot_pos, human_pos
        ) and self._check_robot_facing_human(human_pos, robot_pos):
            found_human = True
            self._val_dict["has_found_human"] = True
            self._val_dict["found_human_times"] += 1

        # Compute the metrics after finding the human
        if self._val_dict["has_found_human"]:
            self._val_dict["dis_after_found"] += dis
            self._val_dict["after_found_human_times"] += found_human

        # Record the step taken to find the human
        if (
            self._val_dict["has_found_human"]
            and self._val_dict["has_found_human_step"] == 1500
        ):
            self._val_dict["has_found_human_step"] = self._val_dict["step"]

        # Compute the minimum distance only when the minimum distance has not found yet
        if (
            self._val_dict["min_start_end_episode_step"] == float("inf")
            and self._enable_shortest_path_computation
        ):
            use_k_human = (
                f"agent_{self._human_idx}_oracle_nav_randcoord_action"
            )
            robot_to_human_min_step = task.actions[
                use_k_human
            ]._compute_robot_to_human_min_step(
                self._robot_init_trans, human_pos, self.human_pos_list
            )

            if robot_to_human_min_step <= self._val_dict["step"]:
                robot_to_human_min_step = self._val_dict["step"]
            else:
                robot_to_human_min_step = float("inf")

            # Update the minimum SPL
            self._val_dict["min_start_end_episode_step"] = min(
                self._val_dict["min_start_end_episode_step"],
                robot_to_human_min_step,
            )

        # Compute the SPL before finding the human
        first_encounter_spl = (
            self._val_dict["has_found_human"]
            * self._val_dict["min_start_end_episode_step"]
            / max(
                self._val_dict["min_start_end_episode_step"],
                self._val_dict["has_found_human_step"],
            )
        )

        # Make sure the first_encounter_spl is not NaN
        if np.isnan(first_encounter_spl):
            first_encounter_spl = 0.0

        self._prev_robot_base_T = mn.Matrix4(
            self._sim.get_agent_data(
                0
            ).articulated_agent.sim_obj.transformation
        )

        # Compute the metrics
        self._metric = {
            "has_found_human": self._val_dict["has_found_human"],
            "found_human_rate_over_epi": self._val_dict["found_human_times"]
            / self._val_dict["step"],
            "found_human_rate_after_encounter_over_epi": self._val_dict[
                "after_found_human_times"
            ]
            / self._val_dict["step_after_found"],
            "avg_robot_to_human_dis_over_epi": self._val_dict["dis"]
            / self._val_dict["step"],
            "avg_robot_to_human_after_encounter_dis_over_epi": self._val_dict[
                "dis_after_found"
            ]
            / self._val_dict["step_after_found"],
            "first_encounter_spl": first_encounter_spl,
            "frist_ecnounter_steps": self._val_dict["has_found_human_step"],
            "frist_ecnounter_steps_ratio": self._val_dict[
                "has_found_human_step"
            ]
            / self._val_dict["min_start_end_episode_step"],
            "follow_human_steps_after_frist_encounter": self._val_dict[
                "after_found_human_times"
            ],
            "follow_human_steps_ratio_after_frist_encounter": self._val_dict[
                "after_found_human_times"
            ]
            / (
                self._total_step - self._val_dict["min_start_end_episode_step"]
            ),
            "backup_ratio": self._val_dict["backup_count"]
            / self._val_dict["step"],
            "yield_ratio": self._val_dict["yield_count"]
            / self._val_dict["step"],
        }

        # Update the counter
        if self._val_dict["has_found_human"]:
            self._val_dict["step_after_found"] += 1


@registry.register_measure
class SocialNavSeekSuccess(Measure):
    """Social nav seek success meassurement"""

    cls_uuid: str = "nav_seek_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return SocialNavSeekSuccess.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        """Reset the metrics"""
        task.measurements.check_measure_dependencies(
            self.uuid,
            [NavToPosSucc.cls_uuid, RotDistToGoal.cls_uuid],
        )
        self._following_step = 0
        self.update_metric(*args, task=task, **kwargs)

    def __init__(self, *args, config, sim, **kwargs):
        self._config = config
        self._sim = sim

        super().__init__(*args, config=config, **kwargs)
        # Setup the parameters
        self._following_step = 0
        self._following_step_succ_threshold = (
            config.following_step_succ_threshold
        )
        self._safe_dis_min = config.safe_dis_min
        self._safe_dis_max = config.safe_dis_max
        self._use_geo_distance = config.use_geo_distance
        self._need_to_face_human = config.need_to_face_human
        self._facing_threshold = config.facing_threshold
        self._robot_idx = config.robot_idx
        self._human_idx = config.human_idx

    def update_metric(self, *args, episode, task, observations, **kwargs):
        # Get the angle distance
        angle_dist = task.measurements.measures[
            RotDistToGoal.cls_uuid
        ].get_metric()

        # Get the positions of the human and the robot
        use_k_human = f"agent_{self._human_idx}_localization_sensor"
        human_pos = observations[use_k_human][:3]
        use_k_robot = f"agent_{self._robot_idx}_localization_sensor"
        robot_pos = observations[use_k_robot][:3]

        # If we want to use the geo distance
        if self._use_geo_distance:
            dist = self._sim.geodesic_distance(robot_pos, human_pos)
        else:
            dist = task.measurements.measures[DistToGoal.cls_uuid].get_metric()

        # Compute facing to human
        base_T = self._sim.get_agent_data(
            0
        ).articulated_agent.base_transformation
        if self._need_to_face_human:
            facing = (
                robot_human_vec_dot_product(robot_pos, human_pos, base_T)
                > self._facing_threshold
            )
        else:
            facing = True

        # Check if the agent follows the human within the safe distance
        if dist >= self._safe_dis_min and dist < self._safe_dis_max and facing:
            self._following_step += 1

        nav_pos_succ = False
        if self._following_step >= self._following_step_succ_threshold:
            nav_pos_succ = True

        # If the robot needs to look at the target
        if self._config.must_look_at_targ:
            self._metric = (
                nav_pos_succ and angle_dist < self._config.success_angle_dist
            )
        else:
            self._metric = nav_pos_succ


@registry.register_sensor
class HumanoidDetectorSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._human_id = config.human_id
        self._human_pixel_threshold = config.human_pixel_threshold
        self._return_image = config.return_image
        self._is_return_image_bbox = config.is_return_image_bbox

        # Check the observation size
        arm_panoptic_shape = None
        head_depth_shape = None
        for key in self._sim.sensor_suite.observation_spaces.spaces:
            if "articulated_agent_arm_panoptic" in key:
                arm_panoptic_shape = (
                    self._sim.sensor_suite.observation_spaces.spaces[key].shape
                )
            if "head_depth" in key:
                head_depth_shape = (
                    self._sim.sensor_suite.observation_spaces.spaces[key].shape
                )

        # Set the correct size
        if arm_panoptic_shape is not None:
            self._height = arm_panoptic_shape[0]
            self._width = arm_panoptic_shape[1]
        else:
            self._height = head_depth_shape[0]
            self._width = head_depth_shape[1]
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return "humanoid_detector_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        if config.return_image:
            return spaces.Box(
                shape=(
                    self._height,
                    self._width,
                    1,
                ),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32,
            )
        else:
            return spaces.Box(
                shape=(1,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32,
            )

    def _get_bbox(self, img):
        """Simple function to get the bounding box, assuming that only one object of interest in the image"""
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def get_observation(self, observations, episode, *args, **kwargs):
        found_human = False
        use_k = f"agent_{self.agent_id}_articulated_agent_arm_panoptic"
        if use_k in observations:
            panoptic = observations[use_k]
        else:
            if self._return_image:
                return np.zeros(
                    (self._height, self._width, 1), dtype=np.float32
                )
            else:
                return np.zeros(1, dtype=np.float32)

        if self._return_image:
            tgt_mask = np.float32(panoptic == self._human_id)
            if self._is_return_image_bbox:
                # Get the bounding box
                bbox = np.zeros(tgt_mask.shape)
                if np.sum(tgt_mask) != 0:
                    rmin, rmax, cmin, cmax = self._get_bbox(tgt_mask)
                    bbox[rmin:rmax, cmin:cmax] = 1.0
                return np.float32(bbox)
            else:
                return tgt_mask
        else:
            if (
                np.sum(panoptic == self._human_id)
                > self._human_pixel_threshold
            ):
                found_human = True

            if found_human:
                return np.ones(1, dtype=np.float32)
            else:
                return np.zeros(1, dtype=np.float32)


@registry.register_sensor
class InitialGpsCompassSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Get the relative distance to the initial starting location of the robot
    """

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return "initial_gps_compass_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(2,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, task, *args, **kwargs):
        agent_data = self._sim.get_agent_data(self.agent_id).articulated_agent
        agent_pos = np.array(agent_data.base_pos)
        init_articulated_agent_T = task.initial_robot_trans

        # Do not support human relative GPS
        if init_articulated_agent_T is None or isinstance(
            agent_data, KinematicHumanoid
        ):
            return np.zeros(2, dtype=np.float32)
        else:
            rel_pos = batch_transform_point(
                np.array([agent_pos]),
                init_articulated_agent_T.inverted(),
                np.float32,
            )
            rho, phi = cartesian_to_polar(rel_pos[0][0], rel_pos[0][1])
            init_rel_pos = np.array([rho, -phi], dtype=np.float32)

            return init_rel_pos

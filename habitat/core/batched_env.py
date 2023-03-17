#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
from collections import OrderedDict
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import magnum as mn
import numpy as np
from gym import spaces
from gym.spaces import Box

from habitat.core.logging import logger
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils import profiling_wrapper
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_sim import build_type
from habitat_sim._ext.habitat_sim_bindings import get_spherical_coordinates
from habitat_sim.utils.common import quat_from_magnum

import torch  # isort:skip # noqa: F401  must import torch before importing bps_pytorch

use_batch_dones_rewards_resets = True


class Camera:  # noqa: SIM119
    def __init__(self, attach_link_name, pos, rotation, hfov):
        self._attach_link_name = attach_link_name
        self._pos = pos
        self._rotation = rotation
        self._hfov = hfov


class StateSensorConfig:
    def __init__(
        self, shape, obs_key, max_episode_length, polar=False, **kwargs
    ):
        self.polar = polar
        self.shape = shape
        self.obs_key = obs_key
        self.max_episode_length = max_episode_length

    def get_obs(self, state) -> np.ndarray:
        raise NotImplementedError()

    # This is for reference only. Beware it is slow!
    def _get_batch_cartesian_coordinates_from_states(self, states):
        robot_positions = np.empty((len(states), 3), dtype=np.float32)
        robot_inv_rotation_mats = np.empty(
            (len(states), 3, 3), dtype=np.float32
        )
        start_positions = np.empty((len(states), 3), dtype=np.float32)
        for i in range(len(states)):
            robot_positions[i] = states[i].robot_pos
            m = states[i].robot_rotation.to_matrix()
            for row in range(3):
                for col in range(3):
                    robot_inv_rotation_mats[i][row][col] = m[col][
                        row
                    ]  # transpose
            start_positions[i] = states[i].robot_start_pos
        return self._get_batch_cartesian_coordinates(
            robot_positions, start_positions, robot_inv_rotation_mats
        )

    def _get_batch_cartesian_coordinates(
        self, source_positions, goal_positions, source_inv_rotation_mats
    ):
        tmp = goal_positions - source_positions
        new_list = np.matmul(
            np.expand_dims(tmp, axis=1), source_inv_rotation_mats
        ).squeeze(axis=1)
        return torch.tensor(new_list)

    def _get_relative_coordinate(
        self, source_position, goal_position, source_rotation
    ):
        if self.polar:
            return _get_spherical_coordinates(
                source_position, goal_position, source_rotation
            )
        else:
            return _get_cartesian_coordinates(
                source_position, goal_position, source_rotation
            )

    def get_batch_obs(self, states, batch_states):
        new_list = np.empty((len(states), self.shape), dtype=np.float32)
        for i in range(len(states)):
            item = self.get_obs(states[i])
            for j in range(self.shape):
                new_list[i][j] = item[j]
        return torch.tensor(new_list)


def _get_cartesian_coordinates(
    source_position, goal_position, source_rotation
):
    # source_T = mn.Matrix4.from_(source_rotation.to_matrix(), source_position)
    # inverted_source_T = source_T.inverted()
    # rel_pos = inverted_source_T.transform_point(goal_position)
    # return np.array(rel_pos)
    return np.array(
        source_rotation.inverted().transform_vector(
            goal_position - source_position
        )
    )


def _get_spherical_coordinates_ref(
    source_position, goal_position, source_rotation
):
    direction_vector = goal_position - source_position
    source_rotation = quat_from_magnum(source_rotation)
    direction_vector_agent = quaternion_rotate_vector(
        # source_rotation.inverse(), direction_vector
        source_rotation.inverse(),
        direction_vector,
    )
    _, phi = cartesian_to_polar(
        -direction_vector_agent[2], direction_vector_agent[0]
    )
    theta = np.arccos(
        direction_vector_agent[1] / np.linalg.norm(direction_vector_agent)
    )
    rho = np.linalg.norm(direction_vector_agent)
    if direction_vector.length() == 0.0:
        # The source is at the same place as the target, theta cannot be calculated
        theta = 0.0
    return rho, theta, phi


def test_get_spherical_coordinates():

    for _ in range(10000):

        source_pos = mn.Vector3(
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
        )
        goal_pos = mn.Vector3(
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
        )
        source_rotation = mn.Quaternion.rotation(
            mn.Deg(np.random.uniform(0, 360)), mn.Vector3(0, 1, 0)
        )

        result0 = _get_spherical_coordinates_ref(
            source_pos, goal_pos, source_rotation
        )
        result1 = get_spherical_coordinates(
            source_pos, goal_pos, source_rotation
        )

        assert not (
            np.isnan(result1.x) or np.isnan(result1.y) or np.isnan(result1.z)
        )
        dist = np.linalg.norm(result1 - result0)
        assert dist < 1e-3


def _get_spherical_coordinates(
    source_position, goal_position, source_rotation
):
    return get_spherical_coordinates(
        source_position, goal_position, source_rotation
    )
    # return _get_spherical_coordinates_ref(source_position, goal_position, source_rotation)


class RobotStartSensorConfig(StateSensorConfig):
    def get_batch_obs(self, states, batch_states):
        assert not self.polar
        return self._get_batch_cartesian_coordinates(
            batch_states.robot_pos,
            batch_states.robot_start_pos,
            batch_states.robot_inv_rotation,
        )

    def get_obs(self, state):
        robot_pos = state.robot_pos
        robot_rot = state.robot_rotation
        start_pos = state.robot_start_pos
        return self._get_relative_coordinate(robot_pos, start_pos, robot_rot)


class RobotTargetSensorConfig(StateSensorConfig):
    def get_batch_obs(self, states, batch_states):
        assert not self.polar
        return self._get_batch_cartesian_coordinates(
            batch_states.robot_pos,
            batch_states.target_obj_start_pos,
            batch_states.robot_inv_rotation,
        )

    def get_obs(self, state):
        robot_pos = state.robot_pos
        robot_rot = state.robot_rotation
        target_pos = state.target_obj_start_pos
        return self._get_relative_coordinate(robot_pos, target_pos, robot_rot)


class RobotGoalSensorConfig(StateSensorConfig):
    def get_batch_obs(self, states, batch_states):
        assert not self.polar
        return self._get_batch_cartesian_coordinates(
            batch_states.robot_pos,
            batch_states.goal_pos,
            batch_states.robot_inv_rotation,
        )

    def get_obs(self, state):
        robot_pos = state.robot_pos
        robot_rot = state.robot_rotation
        target_pos = state.goal_pos
        return self._get_relative_coordinate(robot_pos, target_pos, robot_rot)


class EEStartSensorConfig(StateSensorConfig):
    def get_batch_obs(self, states, batch_states):
        assert not self.polar
        return self._get_batch_cartesian_coordinates(
            batch_states.ee_pos,
            batch_states.robot_start_pos,
            batch_states.ee_inv_rotation,
        )

    def get_obs(self, state):
        ee_pos = state.ee_pos
        ee_rot = state.ee_rotation
        start_pos = state.robot_start_pos
        return self._get_relative_coordinate(ee_pos, start_pos, ee_rot)


class EETargetSensorConfig(StateSensorConfig):
    def get_batch_obs(self, states, batch_states):
        assert not self.polar
        return self._get_batch_cartesian_coordinates(
            batch_states.ee_pos,
            batch_states.target_obj_start_pos,
            batch_states.ee_inv_rotation,
        )

    def get_obs(self, state):
        ee_pos = state.ee_pos
        ee_rot = state.ee_rotation
        target_pos = state.target_obj_start_pos
        return self._get_relative_coordinate(ee_pos, target_pos, ee_rot)


class EEGoalSensorConfig(StateSensorConfig):
    def get_batch_obs(self, states, batch_states):
        assert not self.polar
        return self._get_batch_cartesian_coordinates(
            batch_states.ee_pos,
            batch_states.goal_pos,
            batch_states.ee_inv_rotation,
        )

    def get_obs(self, state):
        ee_pos = state.ee_pos
        ee_rot = state.ee_rotation
        target_pos = state.goal_pos
        return self._get_relative_coordinate(ee_pos, target_pos, ee_rot)


class RobotEESensorConfig(StateSensorConfig):
    def get_batch_obs(self, states, batch_states):
        assert not self.polar
        return self._get_batch_cartesian_coordinates(
            batch_states.robot_pos,
            batch_states.ee_pos,
            batch_states.robot_inv_rotation,
        )

    def get_obs(self, state):
        robot_pos = state.robot_pos
        robot_rot = state.robot_rotation
        ee_pos = state.ee_pos
        return self._get_relative_coordinate(robot_pos, ee_pos, robot_rot)


class JointSensorConfig(StateSensorConfig):
    def get_batch_obs(self, states, batch_states):
        retval = batch_states.robot_joint_positions[:, -9:-2]
        return retval.astype(np.float32)

    def get_obs(self, state):
        return state.robot_joint_positions[-9:-2]


class HoldingSensorConfig(StateSensorConfig):
    def get_batch_obs(self, states, batch_states):
        return np.expand_dims(
            np.not_equal(batch_states.held_obj_idx, -1), 1
        ).astype(np.float32)

    def get_obs(self, state):
        return (float(state.held_obj_idx != -1),)


class StepCountSensorConfig(StateSensorConfig):
    def get_batch_obs(self, states, batch_states):
        fraction_steps_left = (
            (self.max_episode_length - batch_states.episode_step_idx)
            * 1.0
            / self.max_episode_length
        )
        arrays = (
            np.minimum(1.0, fraction_steps_left),
            np.minimum(1.0, fraction_steps_left * 5.0),
            np.minimum(1.0, fraction_steps_left * 50.0),
        )
        retval = np.stack(arrays, axis=1).astype(np.float32)
        return retval

    def get_obs(self, state):
        fraction_steps_left = (
            (self.max_episode_length - state.episode_step_idx)
            * 1.0
            / self.max_episode_length
        )
        return (
            min(1.0, fraction_steps_left * 1.0),
            min(1.0, fraction_steps_left * 5.0),
            min(1.0, fraction_steps_left * 50.0),
        )


# Note this class must match an API expected by PPOTrainer (episode_id and scene_id).
# Beware adding extra state to this class as we often make deep copies of these objects.
class EnvironmentEpisodeState:
    def __init__(self, episode_id):
        self.episode_id = episode_id

    @property
    def scene_id(self):
        return 0  # unused

    def is_disabled(self):
        return self.episode_id == -1

    def set_disabled(self):
        self.episode_id = -1


class BatchedEnv:
    r"""Todo"""

    # observation_spaces: List[spaces.Dict]
    # number_of_episodes: List[Optional[int]]
    # action_spaces: List[spaces.Dict]
    _num_envs: int
    _auto_reset_done: bool

    def __init__(
        self,
        config,
        auto_reset_done: bool = True,
    ) -> None:
        """Todo"""
        self._is_closed = True
        assert config.BATCHED_ENV

        assert (
            config.NUM_ENVIRONMENTS > 0
        ), "number of environments to be created should be greater than 0"

        if not config.get("BLIND", False):
            include_depth = "DEPTH_SENSOR" in config.SENSORS
            include_rgb = "RGB_SENSOR" in config.SENSORS
        else:
            include_depth = False
            include_rgb = False

        self._max_episode_length = config.MAX_EPISODE_LENGTH

        self.state_sensor_config: List[StateSensorConfig] = []
        for ssc_name in config.SENSORS:
            if "RGB" in ssc_name or "DEPTH" in ssc_name:
                continue
            ssc_cfg = config.STATE_SENSORS[ssc_name]
            ssc_cls = eval(ssc_cfg.TYPE)
            ssc = ssc_cls(
                max_episode_length=self._max_episode_length, **ssc_cfg
            )
            self.state_sensor_config.append(ssc)

        assert (include_depth or include_rgb) or config.get("BLIND", False)

        self._num_envs = config.NUM_ENVIRONMENTS

        self._auto_reset_done = auto_reset_done
        self._config = config

        SIMULATOR_GPU_ID = self._config.SIMULATOR_GPU_ID
        sensor_0_name = "HEAD_RGB_SENSOR"
        agent_0_sensor_0_config = getattr(config.SIMULATOR, sensor_0_name)
        sensor0_width, sensor0_height = (
            agent_0_sensor_0_config.WIDTH,
            agent_0_sensor_0_config.HEIGHT,
        )
        depth_sensor_name = "HEAD_DEPTH_SENSOR"
        self.depth_sensor_config = getattr(config.SIMULATOR, depth_sensor_name)
        assert (
            self.depth_sensor_config.WIDTH == sensor0_width
            and self.depth_sensor_config.HEIGHT == sensor0_height
        )
        include_debug_sensor = config.get(
            "EVALUATION_MODE", False
        )  # "DEBUG_RGB_SENSOR" in config.SENSORS
        debug_width = 512
        debug_height = 512

        if not config.STUB_BATCH_SIMULATOR:
            # require CUDA 11.0+ (lower versions will work but runtime perf will be bad!)
            assert torch.cuda.is_available() and torch.version.cuda.startswith(
                "11"
            )
            # you can disable this assert if you really need to test the debug build
            if not config.get("DEBUG_SIM", False):
                assert (
                    build_type == "release"
                ), "Ensure habitat-sim release build for training!"
            from habitat_sim._ext.habitat_sim_bindings import (
                BatchedSimulator,
                BatchedSimulatorConfig,
                EpisodeGeneratorConfig,
            )

            generator_config = EpisodeGeneratorConfig()
            # defaults:
            generator_config.num_episodes = self._config.get(
                "NUM_EPISODES", 100
            )
            # generator_config.seed = 3
            # # # generator_config.num_stage_variations = 12
            # generator_config.min_stage_number = (
            #     70 if config.get("EVALUATION_MODE", False) else 0
            # )
            # generator_config.max_stage_number = (
            #     83 if config.get("EVALUATION_MODE", False) else 69
            # )
            # 0 to 8 is training setup and 9 to 11 is testing setup

            # generator_config.num_object_variations = 6
            generator_config.min_nontarget_objects = self._config.get(
                "MIN_NON_TARGET", 27
            )
            generator_config.max_nontarget_objects = self._config.get(
                "MAX_NON_TARGET", 32
            )
            generator_config.use_fixed_robot_start_pos = self._config.get(
                "FIXED_ROBOT_START_POSITION", True
            )
            # generator_config.use_fixed_robot_start_yaw = False
            # generator_config.use_fixed_robot_joint_start_positions = False

            bsim_config = BatchedSimulatorConfig()
            bsim_config.gpu_id = SIMULATOR_GPU_ID
            bsim_config.include_depth = include_depth
            bsim_config.include_color = include_rgb
            bsim_config.num_envs = self._num_envs
            bsim_config.sensor0.width = sensor0_width
            bsim_config.sensor0.height = sensor0_height
            bsim_config.num_debug_envs = (
                self._num_envs if include_debug_sensor else 0
            )
            bsim_config.debug_sensor.width = debug_width
            bsim_config.debug_sensor.height = debug_height
            bsim_config.force_random_actions = False
            bsim_config.do_async_physics_step = self._config.OVERLAP_PHYSICS
            bsim_config.enable_sliding = self._config.get(
                "ENABLE_SLIDING", False
            )
            bsim_config.num_physics_substeps = (
                self._config.NUM_PHYSICS_SUBSTEPS
            )
            generate_episodes = self._config.get("PROCEDURAL_GENERATION", True)
            bsim_config.do_procedural_episode_set = generate_episodes
            if not generate_episodes:
                if config.get("EVALUATION_MODE", False):
                    bsim_config.episode_set_filepath = self._config[
                        "EVAL_DATASET"
                    ]
                else:
                    bsim_config.episode_set_filepath = self._config[
                        "TRAIN_DATASET"
                    ]
            bsim_config.episode_generator_config = generator_config

            bsim_config.enable_robot_collision = self._config.get(
                "ENABLE_ROBOT_COLLISION", True
            )
            bsim_config.enable_held_object_collision = self._config.get(
                "ENABLE_HELD_OBJECT_COLLISION", True
            )

            self._bsim = BatchedSimulator(bsim_config)

            self.action_dim = self._bsim.get_num_actions()

            self._bsim.enable_debug_sensor(False)

            camera_setting = self._config.get("CAMERA_CONFIG", "DEFAULT")
            if camera_setting == "DEFAULT":
                self._main_camera = Camera(
                    "head_pan_link",
                    mn.Vector3(0.2, 0.2, 0.0),
                    mn.Quaternion.rotation(
                        mn.Deg(-90.0), mn.Vector3(0.0, 1.0, 0.0)
                    )
                    * mn.Quaternion.rotation(
                        mn.Deg(-20.0), mn.Vector3(1.0, 0.0, 0.0)
                    ),
                    60,
                )
                self.set_camera("sensor0", self._main_camera)
            elif camera_setting == "TRANSFER_1":
                self._main_camera = Camera(
                    "head_pan_link",
                    mn.Vector3(0.2, 0.2, 0.0),
                    mn.Quaternion.rotation(
                        mn.Deg(-90.0), mn.Vector3(0.0, 1.0, 0.0)
                    )
                    * mn.Quaternion.rotation(
                        mn.Deg(-20.0), mn.Vector3(1.0, 0.0, 0.0)
                    ),
                    90,
                )
                self.set_camera("sensor0", self._main_camera)
            elif camera_setting == "TRANSFER_2":
                self._main_camera = Camera(
                    "head_pan_link",
                    mn.Vector3(0.2, 0.0, 0.0),
                    mn.Quaternion.rotation(
                        mn.Deg(-90.0), mn.Vector3(0.0, 1.0, 0.0)
                    )
                    * mn.Quaternion.rotation(
                        mn.Deg(-20.0), mn.Vector3(1.0, 0.0, 0.0)
                    ),
                    90,
                )
                self.set_camera("sensor0", self._main_camera)
            else:
                raise NotImplementedError(
                    f"Unknown camera setting {camera_setting}"
                )
            if include_debug_sensor:
                self._debug_camera = Camera(
                    "base_link",
                    mn.Vector3(
                        -0.8, 2.7, -0.8
                    ),  # place behind, above, and to the left of the base
                    mn.Quaternion.rotation(
                        mn.Deg(-120.0), mn.Vector3(0.0, 1.0, 0.0)
                    )  # face 30 degs to the right
                    * mn.Quaternion.rotation(
                        mn.Deg(-65.0), mn.Vector3(1.0, 0.0, 0.0)
                    ),  # tilt down
                    60,
                )
                self.set_camera("debug", self._debug_camera)

            # reference code for wide-angle camera
            # self._main_camera = Camera("torso_lift_link",
            #     mn.Vector3(-0.536559, 2.16173, 0.568379),
            #     mn.Quaternion(mn.Vector3(-0.26714, -0.541109, -0.186449), 0.775289),
            #     75)

            self.is_eval = False

        else:
            self._bsim = None
            self.action_dim = 10  # arbitrary

        buffer_index = 0

        self._raw_rgb = None
        self._raw_depth = None
        self._raw_debug_rgb = None
        assert self._bsim
        import bps_pytorch  # see https://github.com/shacklettbp/bps-nav#building

        if include_rgb:
            self._raw_rgb = bps_pytorch.make_color_tensor(
                self._bsim.rgba(buffer_index),
                SIMULATOR_GPU_ID,
                self._num_envs,
                [sensor0_height, sensor0_width],
            )[..., 0:3]

        if include_depth:
            self._raw_depth = bps_pytorch.make_depth_tensor(
                self._bsim.depth(buffer_index),
                SIMULATOR_GPU_ID,
                self._num_envs,
                [sensor0_height, sensor0_width],
            )

        if include_debug_sensor:
            self._raw_debug_rgb = bps_pytorch.make_color_tensor(
                self._bsim.debug_rgba(buffer_index),
                SIMULATOR_GPU_ID,
                self._num_envs,
                [debug_height, debug_width],
            )[..., 0:3]

        self._observations: OrderedDict = OrderedDict()

        self._is_closed = False

        obs_dict = spaces.Dict({})
        if include_rgb:
            RGBSENSOR_DIMENSION = 3
            rgb_obs = spaces.Box(
                low=0,
                high=255,
                shape=(
                    agent_0_sensor_0_config.HEIGHT,
                    agent_0_sensor_0_config.WIDTH,
                    RGBSENSOR_DIMENSION,
                ),
                dtype=np.uint8,
            )
            obs_dict["rgb"] = rgb_obs

        if include_depth:
            if self.depth_sensor_config.NORMALIZE_DEPTH:
                min_depth_value = 0
                max_depth_value = 1
            else:
                min_depth_value = config.MIN_DEPTH
                max_depth_value = config.MAX_DEPTH

            depth_obs = spaces.Box(
                low=min_depth_value,
                high=max_depth_value,
                shape=(
                    agent_0_sensor_0_config.HEIGHT,
                    agent_0_sensor_0_config.WIDTH,
                    1,
                ),
                dtype=np.float32,
            )
            obs_dict["depth"] = depth_obs

        if include_debug_sensor:
            RGBSENSOR_DIMENSION = 3
            debug_rgb_obs = spaces.Box(
                low=0,
                high=255,
                shape=(
                    debug_height,
                    debug_width,
                    RGBSENSOR_DIMENSION,
                ),
                dtype=np.uint8,
            )
            obs_dict["debug_rgb"] = debug_rgb_obs

        for ssc in self.state_sensor_config:
            obs_dict[ssc.obs_key] = spaces.Box(
                low=-np.inf,
                high=np.inf,  # todo: investigate depth min/max
                shape=(ssc.shape,),
                dtype=np.float32,
            )

        self.observation_spaces = [
            obs_dict
        ] * 1  # config.NUM_ENVIRONMENTS  # note we only ever read element #0 of this array

        action_space = Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        self.rewards = np.zeros((self._num_envs), dtype=float)
        self.dones = np.zeros((self._num_envs), dtype=bool)
        self.infos: List[Dict[str, Any]] = [{}] * self._num_envs
        self._previous_state: List[Optional[Any]] = [None] * self._num_envs
        self._previous_action: List[Optional[Any]] = [None] * self._num_envs
        self._previous_actions = np.zeros(
            (self._num_envs, self.action_dim), dtype=float
        )
        self._past_pick_success = np.zeros((self._num_envs), dtype=bool)
        self._drop_position_arr = np.zeros((self._num_envs, 3), dtype=float)
        self._drop_position_single = [None] * self._num_envs
        self._has_drop_position = np.zeros((self._num_envs), dtype=bool)
        self._object_dropped_properly = np.zeros((self._num_envs), dtype=bool)
        self._stagger_agents = np.zeros((self._num_envs), dtype=int)
        self._previous_action_for_penalty = np.zeros(
            (self._num_envs, self.action_dim), dtype=float
        )
        if self._config.get("STAGGER", True):
            self._stagger_agents = np.array(
                [i % self._max_episode_length for i in range(self._num_envs)]
            )

        self.action_spaces = [
            action_space
        ] * 1  # note we only ever read element #0 of this array
        # self.number_of_episodes = []
        self._paused: List[int] = []
        self._num_episodes = self._bsim.get_num_episodes()
        self._next_episode_idx = 0

    def set_camera(self, sensor_name, camera):
        self._bsim.set_camera(
            sensor_name,
            camera._pos,
            camera._rotation,
            camera._hfov,
            camera._attach_link_name,
        )

    @property
    def num_envs(self):
        r"""number of individual environments."""
        return self._num_envs

    @property
    def number_of_episodes(self):
        return [self._num_episodes]  # user code wants a list of counts

    def get_next_episode(self):
        assert self._num_episodes > 0
        retval = self._next_episode_idx

        if self.is_eval:
            # for eval, we launch all episodes once, then return -1
            if self._next_episode_idx == -1:
                pass
            elif self._next_episode_idx + 1 == self._num_episodes:
                self._next_episode_idx = -1
            else:
                self._next_episode_idx += 1
        else:
            self._next_episode_idx = (
                self._next_episode_idx + 1
            ) % self._num_episodes

        return retval

    def current_episodes(self):
        # make deep copy of _current_episodes
        return copy.deepcopy(self._current_episodes)

    def count_episodes(self):
        raise NotImplementedError()

    def episode_over(self):
        raise NotImplementedError()

    def get_metrics(self):
        raise NotImplementedError()

    def get_nonpixel_observations(self, env_states, observations):
        # for (b, state) in enumerate(env_states):
        #     for ssc in self.state_sensor_config:
        #         sensor_data = torch.tensor(ssc.get_obs(state))
        #         observations[ssc.obs_key][b, :] = sensor_data

        batch_env_states_curr = self._bsim.get_batch_environment_state(
            previous=False
        )
        # batch_env_states_prev = self._bsim.get_batch_environment_state(previous=True)

        for ssc in self.state_sensor_config:
            observations[ssc.obs_key] = torch.tensor(
                ssc.get_batch_obs(env_states, batch_env_states_curr)
            )
            if not torch.isfinite(observations[ssc.obs_key]).all():
                logger.info((ssc.obs_key, "nan"))
                for b, s in enumerate(env_states):
                    logger.info(observations[ssc.obs_key][b, :])
                    logger.info(
                        (
                            s.robot_pos,
                            s.robot_rotation,
                            s.robot_start_pos,
                            s.robot_start_rotation,
                            s.ee_pos,
                            s.ee_rotation,
                            s.robot_start_pos,
                            s.robot_start_rotation,
                            s.target_obj_start_pos,
                        )
                    )
                observations[ssc.obs_key] = torch.nan_to_num(
                    observations[ssc.obs_key], 0.0
                )

    def get_batch_dones_rewards_resets(self, actions):
        state = self._bsim.get_batch_environment_state(previous=False)
        prev_state = self._bsim.get_batch_environment_state(previous=True)
        return self.get_batch_dones_rewards_resets_helper(
            state, prev_state, actions
        )

    @profiling_wrapper.RangeContext("get_batch_dones_rewards_resets")
    def get_batch_dones_rewards_resets_helper(
        self, state, prev_state, actions
    ):

        has_valid_prev_state = np.where(state.episode_step_idx, True, False)

        config_action_penalty = self._config.get("ACTION_PENALTY", 0.0)
        config_end_action_threshold = self._config.get(
            "END_ACTION_THRESHOLD", 0.0
        )
        config_drop_threshold = self._config.get("DROP_THRESHOLD", 0.01)
        config_grasp_threshold = self._config.get("GRASP_THRESHOLD", 0.02)
        drop_grasp_action_idx = 0
        end_action_index = self.action_dim - 1

        np_actions = actions.cpu().numpy()

        continuous_action_norm = np_actions.clip(-1, 1)

        continuous_action_norm[
            :, drop_grasp_action_idx
        ] = 0.0  # no penalty on the grasp action
        continuous_action_norm[
            :, end_action_index
        ] = 0.0  # no penalty on the end action

        if self._config.get("ACTION_PENALTY_ON_BASE", False):
            continuous_action_norm[:, 1] = 0.0
            continuous_action_norm[:, 2] = 0.0

        # continuous_action_l2 = sum(c * c for c in continuous_action_norm)
        # continuous_action_l2 = np.sum(np.dot(continuous_action_norm))
        if self._config.get("ACTION_PENALTY_ON_DIFF", False):
            action_diff = (
                self._previous_action_for_penalty - continuous_action_norm
            )
            continuous_action_l2 = np.sum(action_diff * action_diff, axis=1)
            action_penalty = config_action_penalty * continuous_action_l2
            self._previous_action_for_penalty = continuous_action_norm
        else:
            continuous_action_l2 = np.sum(
                continuous_action_norm * continuous_action_norm, axis=1
            )
            action_penalty = config_action_penalty * continuous_action_l2
        # continuous_action_norm_mean = sum(
        #     abs(c) for c in continuous_action_norm
        # ) / len(continuous_action_norm)
        # continuous_action_norm_median = median(
        #     abs(c) for c in continuous_action_norm
        # )

        end_episode_action = (
            np_actions[:, end_action_index] > config_end_action_threshold
        )

        original_drop_grasp = np_actions[:, drop_grasp_action_idx].copy()

        modified_drop_grasp = np.where(
            original_drop_grasp < config_drop_threshold, -1.0, 0.0
        )
        modified_drop_grasp = np.where(
            original_drop_grasp > config_grasp_threshold,
            1.0,
            modified_drop_grasp,
        )

        np_actions[:, drop_grasp_action_idx] = modified_drop_grasp
        if self._config.get("LOCK_BASE", False):
            np_actions[:, 1] = 0
            np_actions[:, 2] = 0

        end_episode_action = np.logical_and(
            end_episode_action, np.greater(state.episode_step_idx, 5)
        )

        # note self._previous_action and tried_grasp_last_step are broken in main:
        # _previous_action is always None and tried_grasp_last_step is always False.
        # This code preserves the broken behavior.
        # tried_grasp_last_step = np.zeros_like(has_valid_prev_state, dtype=bool)

        # tried_grasp_last_step = (
        #     self._previous_action[b] is not None
        # ) and (self._previous_action[b][(b * self.action_dim)] > 0.0)
        # tried_grasp_last_step = np.logical_and(
        #     has_valid_prev_state,
        #     np.greater(self._previous_actions[:, drop_grasp_action_idx], 0.0),
        # )

        # ee_to_start = (state.target_obj_start_pos - state.ee_pos).length()
        # ee_to_start = np.linalg.norm(
        #     state.target_obj_start_pos - state.ee_pos, axis=1
        # )

        is_holding_correct = state.target_obj_idx == state.held_obj_idx
        is_not_holding_an_object = state.held_obj_idx == -1
        # was_holding_correct = False
        # if prev_state is not None:
        #     was_holding_correct = (
        #         prev_state.target_obj_idx == prev_state.held_obj_idx
        #     )

        # was_holding_correct = np.where(
        #     has_valid_prev_state,
        #     prev_state.target_obj_idx == prev_state.held_obj_idx,
        #     False,
        # )

        # todo: keep porting

        # self._past_pick_success[b] = (
        #     self._past_pick_success[b] or is_holding_correct
        # )
        self._past_pick_success = np.logical_or(
            self._past_pick_success, is_holding_correct
        )

        # obj_pos = state.obj_positions[state.target_obj_idx]
        # obj_to_goal = (state.goal_pos - obj_pos).length()
        obj_pos = state.target_obj_pos
        obj_to_goal = np.linalg.norm(state.goal_pos - obj_pos, axis=1)

        object_is_close_to_goal = np.logical_or(
            np.less(obj_to_goal, self._config.NPNP_SUCCESS_THRESH),
            self._object_dropped_properly,
        )

        bad_attempt_penalty = np.zeros_like(has_valid_prev_state, dtype=float)

        do_set_drop_pos = (
            np.equal(modified_drop_grasp, -1.0)
            & is_holding_correct
            & np.logical_not(self._has_drop_position)
        )

        self._drop_position_arr = np.where(
            np.expand_dims(do_set_drop_pos, axis=1),
            obj_pos,
            self._drop_position_arr,
        )
        self._has_drop_position = np.logical_or(
            do_set_drop_pos, self._has_drop_position
        )

        # todo: port this to tensor ops
        if self._config.get("DO_NOT_END_IF_DROP_WRONG", False):
            assert self._config.get("TASK_HAS_SIMPLE_PLACE", False)

        success = object_is_close_to_goal
        if self._config.get("TASK_HAS_SIMPLE_PLACE", False):
            success = success.fill(False)
            drop_to_goal = self._drop_position_arr - state.goal_pos
            drop_success = (
                np.linalg.norm(drop_to_goal[:, (0, 2)], axis=1)
                < self._config.NPNP_SUCCESS_THRESH
            )
            drop_success &= drop_to_goal[:, 1] > 0
            drop_success &= drop_to_goal[:, 1] < (
                2 * self._config.NPNP_SUCCESS_THRESH
            )
            success = drop_success & self._has_drop_position
            if self._config.get("DO_NOT_END_IF_DROP_WRONG", False):
                self._has_drop_position &= drop_success
                np_actions[:, drop_grasp_action_idx] = np.where(
                    ~drop_success, 1.0, np_actions[:, drop_grasp_action_idx]
                )
                assert self._config.get("DROP_WRONG_PENALTY", 0.0) == 0.0

        if not self._config.get("TASK_NO_END_ACTION", False):
            success = success & end_episode_action

        # todo: port code
        assert not self._config.get(
            "TASK_IS_PICK_ONLY_FAIL_IF_BAD_ATTEMPT", False
        )
        # if self._config.get(
        #     "TASK_IS_PICK_ONLY_FAIL_IF_BAD_ATTEMPT", False
        # ):
        #     success = is_holding_correct and end_episode_action

        # todo: port code
        if self._config.get("TASK_IS_PLACE", False):
            success = np.logical_and(success, is_not_holding_an_object)

        if self._config.get("TASK_IS_NAV_PICK_NAV_REACH", False):
            success = object_is_close_to_goal

        if self._config.get("TASK_IS_SIMPLE_PICK", False):
            success = is_holding_correct

        if self._config.get("PREVENT_STOP_ACTION", False):
            assert self._config.get("TASK_NO_END_ACTION", False)
            #     end_episode_action = False
            end_episode_action.fill(False)

        if self._config.get("DROP_IS_FAIL", True):
            failure = (
                (
                    state.did_drop
                    & (obj_to_goal >= self._config.NPNP_SUCCESS_THRESH)
                )
                | (~is_holding_correct & ~is_not_holding_an_object)
                | end_episode_action
            )
            failure = failure & ~success

        else:
            failure = end_episode_action & ~success

        # todo: port code
        assert not self._config.get(
            "TASK_IS_PICK_ONLY_FAIL_IF_BAD_ATTEMPT", False
        )
        # if self._config.get(
        #     "TASK_IS_PICK_ONLY_FAIL_IF_BAD_ATTEMPT", False
        # ):
        #     failure = (
        #         (not is_holding_correct) and tried_grasp_last_step
        #     ) or state.did_drop

        # if (
        #     success
        #     or failure
        #     or state.episode_step_idx
        #     >= (self._max_episode_length - self._stagger_agents[b])
        # ):
        do_end_episode = np.logical_or(
            np.logical_or(success, failure),
            np.greater_equal(
                state.episode_step_idx,
                self._max_episode_length - self._stagger_agents,
            ),
        )

        # self.resets[b] = -1
        # self.dones[b] = False
        # self.rewards[b] = -self._config.NPNP_SLACK_PENALTY
        self.resets.fill(-1)
        self.dones.fill(False)
        self.rewards.fill(-self._config.NPNP_SLACK_PENALTY)

        # prev_obj_pos = prev_state.obj_positions[
        #     prev_state.target_obj_idx
        # ]
        prev_obj_pos = prev_state.target_obj_pos
        # curr_dist_ee_to_obj = (
        #     state.obj_positions[state.target_obj_idx]
        #     - state.ee_pos
        # ).length()
        curr_dist_ee_to_obj = np.linalg.norm(
            state.target_obj_pos - state.ee_pos, axis=1
        )

        # prev_dist_ee_to_obj = (
        #     prev_obj_pos - prev_state.ee_pos
        # ).length()
        prev_dist_ee_to_obj = np.linalg.norm(
            prev_obj_pos - prev_state.ee_pos, axis=1
        )
        # prev_obj_to_goal = (
        #     prev_state.goal_pos - prev_obj_pos
        # ).length()
        prev_obj_to_goal = np.linalg.norm(
            prev_state.goal_pos - prev_obj_pos, axis=1
        )

        # self.rewards[b] += -(
        #     curr_dist_ee_to_obj - prev_dist_ee_to_obj
        # ) * self._config.get("CARTHESIAN_REWARD", 1.0)
        extra_reward_term = -(
            curr_dist_ee_to_obj - prev_dist_ee_to_obj
        ) * self._config.get("CARTHESIAN_REWARD", 1.0)

        # self.rewards[b] += -(
        #     obj_to_goal - prev_obj_to_goal
        # ) * self._config.get("CARTHESIAN_REWARD", 1.0)
        extra_reward_term += -(
            obj_to_goal - prev_obj_to_goal
        ) * self._config.get("CARTHESIAN_REWARD", 1.0)

        # self.rewards[b] -= bad_attempt_penalty
        # self.rewards[b] -= action_penalty
        extra_reward_term -= bad_attempt_penalty
        extra_reward_term -= action_penalty

        # if (
        #     self._config.get("DROP_IS_FAIL", True)
        #     and is_holding_correct
        #     and (prev_state.held_obj_idx == -1)
        #     and obj_to_goal
        #     > self._config.NPNP_SUCCESS_THRESH  # the robot is not re-picking
        # ):
        do_give_pick_reward = np.logical_and(
            np.logical_and(
                np.logical_and(
                    self._config.get("DROP_IS_FAIL", True), is_holding_correct
                ),
                prev_state.held_obj_idx == -1,
            ),
            obj_to_goal > self._config.NPNP_SUCCESS_THRESH,
        )

        # self.rewards[b] += self._config.PICK_REWARD
        extra_reward_term += np.where(
            do_give_pick_reward, self._config.PICK_REWARD, 0
        )

        self.rewards += np.where(has_valid_prev_state, extra_reward_term, 0.0)

        if self._config.LOG_INFO:
            dist_to_start = np.linalg.norm(
                state.target_obj_start_pos - state.ee_pos, axis=1
            )
        else:
            dist_to_start = None

        for b in range(self.num_envs):
            if self._current_episodes[b].is_disabled():
                # let this episode continue in the sim; ignore the results
                assert self.resets[b] == -1
                self.dones[b] = False
                self.rewards[b] = 0
                self.infos[b] = {}
                continue

            if do_end_episode[b]:
                self._stagger_agents[b] = 0
                self.dones[b] = True
                _rew = 0.0
                _rew += self._config.NPNP_SUCCESS_REWARD if success[b] else 0.0
                _rew -= (
                    self._config.NPNP_FAILURE_PENALTY if failure[b] else 0.0
                )
                _rew -= bad_attempt_penalty[b]
                self.rewards[b] = _rew
                if (
                    self._config.LOG_INFO
                    and b < self._config.LOG_INFO_NUM_ENVS
                ):
                    self.infos[b] = {
                        "success": float(success[b]),
                        "failure": float(failure[b]),
                        "pick_success": float(self._past_pick_success[b]),
                        "episode_steps": state.episode_step_idx[b],
                        "distance_to_start": dist_to_start[b],
                        "distance_to_goal": obj_to_goal[b],
                        # "try_grasp": actions[(b * self.action_dim)]
                        # > self._config.get("GRASP_THRESHOLD", 0.02),
                        # "try_drop": actions[(b * self.action_dim)]
                        # < self._config.get("DROP_THRESHOLD", 0.01),
                        "is_holding_correct": float(is_holding_correct[b]),
                        # "was_holding_correct": float(was_holding_correct),
                        "end_action": float(end_episode_action[b]),
                        # "_continuous_action_norm_mean": continuous_action_norm_mean,
                        # "_continuous_action_norm_median": continuous_action_norm_median,
                        # "_continuous_action_l2": continuous_action_l2,
                        "_original_drop_grasp": float(original_drop_grasp[b]),
                        # "_state.did_grasp": state.did_grasp,
                        # "_state.did_attempt_grasp": state.did_attempt_grasp,
                        # "_state.did_collide": state.did_collide,
                    }
                self._previous_state[b] = None
                self._previous_action[b] = None
                self._past_pick_success[b] = False
                self._drop_position_arr[b] = None
                self._has_drop_position[b] = False
                self._object_dropped_properly[b] = False
                self._previous_action_for_penalty[b].fill(0.0)

                next_episode = self.get_next_episode()
                if next_episode != -1:
                    self.resets[b] = next_episode
                    self._current_episodes[b].episode_id = next_episode
                else:
                    # There are no more episodes to launch, so disable this env. We'll
                    # hit this case during eval.
                    # Note we don't yet communicate a disabled env to the sim; the
                    # sim continues simulating this episode and we'll ignore the result.
                    self.resets[b] = -1  # don't reset env
                    self._current_episodes[b].set_disabled()

            else:
                if (
                    self._config.LOG_INFO
                    and b < self._config.LOG_INFO_NUM_ENVS
                ):
                    self.infos[b] = {
                        "success": float(success[b]),
                        "failure": float(failure[b]),
                        "pick_success": float(self._past_pick_success[b]),
                        "episode_steps": state.episode_step_idx[b],
                        "distance_to_start": dist_to_start[b],
                        "distance_to_goal": obj_to_goal[b],
                        # "try_grasp": actions[(b * self.action_dim)]
                        # > self._config.get("GRASP_THRESHOLD", 0.02),
                        # "try_drop": actions[(b * self.action_dim)]
                        # < self._config.get("DROP_THRESHOLD", 0.01),
                        "is_holding_correct": float(is_holding_correct[b]),
                        # "was_holding_correct": float(was_holding_correct),
                        "end_action": float(end_episode_action[b]),
                        # "_continuous_action_norm_mean": continuous_action_norm_mean,
                        # "_continuous_action_norm_median": continuous_action_norm_median,
                        # "_continuous_action_l2": continuous_action_l2,
                        "_original_drop_grasp": float(original_drop_grasp[b]),
                        # "_state.did_grasp": state.did_grasp,
                        # "_state.did_attempt_grasp": state.did_attempt_grasp,
                        # "_state.did_collide": state.did_collide,
                    }
        return np_actions

    def get_dones_rewards_resets(self, env_states, actions):
        # assert not use_batch_dones_rewards_resets
        for (b, state) in enumerate(env_states):
            if self._current_episodes[b].is_disabled():
                # let this episode continue in the sim; ignore the results
                assert self.resets[b] == -1
                self.dones[b] = False
                self.rewards[b] = 0
                self.infos[b] = {}
                continue

            continuous_action_norm = actions[
                b * self.action_dim : (b + 1) * self.action_dim
            ]
            # continuous_action_norm = actions[
            #     b * self.action_dim + 1 : b * self.action_dim + 10
            # ]
            continuous_action_l2 = sum(c * c for c in continuous_action_norm)
            action_penalty = (
                self._config.get("ACTION_PENALTY", 0.0) * continuous_action_l2
            )
            continuous_action_norm_mean = sum(
                abs(c) for c in continuous_action_norm
            ) / len(continuous_action_norm)
            continuous_action_norm_median = median(
                abs(c) for c in continuous_action_norm
            )

            end_episode_action = actions[
                (b + 1) * self.action_dim - 1
            ] > self._config.get("END_ACTION_THRESHOLD", 0.0)
            original_drop_grasp = actions[b * self.action_dim]
            if actions[b * self.action_dim] < self._config.get(
                "DROP_THRESHOLD", 0.01
            ):
                actions[b * self.action_dim] = -1.0
            elif actions[b * self.action_dim] > self._config.get(
                "GRASP_THRESHOLD", 0.02
            ):
                actions[b * self.action_dim] = 1.0
            else:
                actions[b * self.action_dim] = 0.0

            if self._config.get("LOCK_BASE", False):
                actions[b * self.action_dim + 1] = 0
                actions[b * self.action_dim + 2] = 0

            end_episode_action = (
                end_episode_action and state.episode_step_idx > 5
            )

            tried_grasp_last_step = (
                self._previous_action[b] is not None
            ) and (self._previous_action[b][(b * self.action_dim)] > 0.0)

            prev_state = self._previous_state[b]
            ee_to_start = (state.target_obj_start_pos - state.ee_pos).length()
            # success = curr_dist < self._config.REACH_SUCCESS_THRESH
            # success = state.target_obj_idx == state.held_obj_idx

            is_holding_correct = state.target_obj_idx == state.held_obj_idx
            was_holding_correct = False
            if prev_state is not None:
                was_holding_correct = (
                    prev_state.target_obj_idx == prev_state.held_obj_idx
                )
            self._past_pick_success[b] = (
                self._past_pick_success[b] or is_holding_correct
            )

            obj_pos = state.obj_positions[state.target_obj_idx]
            obj_to_goal = (state.goal_pos - obj_pos).length()

            object_is_close_to_goal = (
                obj_to_goal
                < self._config.NPNP_SUCCESS_THRESH
                # np.sqrt(
                #         (state.goal_pos[0] - obj_pos[0]) ** 2
                #         + (state.goal_pos[2] - obj_pos[2]) ** 2
                #     ) < self._config.NPNP_SUCCESS_THRESH
            ) or self._object_dropped_properly[b]

            if (
                actions[(b * self.action_dim)] == -1.0
                and is_holding_correct
                and self._drop_position_single[b] is None
            ):
                self._drop_position_single[b] = obj_pos

            bad_attempt_penalty = 0.0
            if self._config.get("DO_NOT_END_IF_DROP_WRONG", False):
                assert self._config.get("TASK_HAS_SIMPLE_PLACE", False)
                if self._drop_position_single[b] is not None:
                    drop_to_goal = (
                        self._drop_position_single[b] - state.goal_pos
                    )
                    drop_success = (
                        np.sqrt(drop_to_goal[0] ** 2 + drop_to_goal[2] ** 2)
                        < self._config.NPNP_SUCCESS_THRESH
                    )
                    drop_success = drop_success and drop_to_goal[1] > 0
                    drop_success = (
                        drop_success
                        and drop_to_goal[1]
                        < 2 * self._config.NPNP_SUCCESS_THRESH
                    )
                    if not drop_success:
                        self._drop_position_single[b] = None
                        actions[(b * self.action_dim)] = 1.0
                        bad_attempt_penalty = self._config.get(
                            "DROP_WRONG_PENALTY", 0.0
                        )

            success = object_is_close_to_goal
            if self._config.get("TASK_HAS_SIMPLE_PLACE", False):
                success = False
                if self._drop_position_single[b] is not None:
                    drop_to_goal = (
                        self._drop_position_single[b] - state.goal_pos
                    )
                    drop_success = (
                        np.sqrt(drop_to_goal[0] ** 2 + drop_to_goal[2] ** 2)
                        < self._config.NPNP_SUCCESS_THRESH
                    )
                    drop_success = drop_success and drop_to_goal[1] > 0
                    drop_success = (
                        drop_success
                        and drop_to_goal[1]
                        < 2 * self._config.NPNP_SUCCESS_THRESH
                    )

                    success = drop_success

            if not self._config.get("TASK_NO_END_ACTION", False):
                success = success and end_episode_action

            if self._config.get(
                "TASK_IS_PICK_ONLY_FAIL_IF_BAD_ATTEMPT", False
            ):
                success = is_holding_correct and end_episode_action

            if self._config.get("TASK_IS_PLACE", False):
                success = (
                    success
                    and (state.held_obj_idx == -1)
                    # and self._past_pick_success[b]
                )
            if self._config.get("TASK_IS_NAV_PICK_NAV_REACH", False):
                success = object_is_close_to_goal

            if self._config.get("TASK_IS_SIMPLE_PICK", False):
                success = is_holding_correct

            if self._config.get("PREVENT_STOP_ACTION", False):
                assert self._config.get("TASK_NO_END_ACTION", False)
                end_episode_action = False

            if self._config.get("DROP_IS_FAIL", True):
                failure = (
                    (
                        state.did_drop
                        and (obj_to_goal >= self._config.NPNP_SUCCESS_THRESH)
                    )
                    or (
                        state.target_obj_idx != state.held_obj_idx
                        and state.held_obj_idx != -1
                    )
                    or (end_episode_action)
                )
                failure = failure and not success
            else:
                failure = end_episode_action and not success

            if self._config.get(
                "TASK_IS_PICK_ONLY_FAIL_IF_BAD_ATTEMPT", False
            ):
                failure = (
                    (not is_holding_correct) and tried_grasp_last_step
                ) or state.did_drop

            if (
                success
                or failure
                or state.episode_step_idx
                >= (self._max_episode_length - self._stagger_agents[b])
            ):
                self._stagger_agents[b] = 0
                self.dones[b] = True
                _rew = 0.0
                _rew += self._config.NPNP_SUCCESS_REWARD if success else 0.0
                _rew -= self._config.NPNP_FAILURE_PENALTY if failure else 0.0
                _rew -= bad_attempt_penalty
                self.rewards[b] = _rew
                self.infos[b] = {
                    "success": float(success),
                    "failure": float(failure),
                    "pick_success": float(self._past_pick_success[b]),
                    "episode_steps": state.episode_step_idx,
                    "distance_to_start": ee_to_start,
                    "distance_to_goal": obj_to_goal,
                    "try_grasp": actions[(b * self.action_dim)]
                    > self._config.get("GRASP_THRESHOLD", 0.02),
                    "try_drop": actions[(b * self.action_dim)]
                    < self._config.get("DROP_THRESHOLD", 0.01),
                    "is_holding_correct": float(is_holding_correct),
                    "was_holding_correct": float(was_holding_correct),
                    "end_action": float(end_episode_action),
                    "_continuous_action_norm_mean": continuous_action_norm_mean,
                    "_continuous_action_norm_median": continuous_action_norm_median,
                    "_continuous_action_l2": continuous_action_l2,
                    "_original_drop_grasp": original_drop_grasp,
                    "_state.did_grasp": state.did_grasp,
                    "_state.did_attempt_grasp": state.did_attempt_grasp,
                    "_state.did_collide": state.did_collide,
                }
                self._previous_state[b] = None
                self._previous_action[b] = None
                self._past_pick_success[b] = False
                self._drop_position_single[b] = None
                self._object_dropped_properly[b] = False

                next_episode = self.get_next_episode()
                if next_episode != -1:
                    self.resets[b] = next_episode
                    self._current_episodes[b].episode_id = next_episode
                else:
                    # There are no more episodes to launch, so disable this env. We'll
                    # hit this case during eval.
                    # Note we don't yet communicate a disabled env to the sim; the
                    # sim continues simulating this episode and we'll ignore the result.
                    self.resets[b] = -1  # don't reset env
                    self._current_episodes[b].set_disabled()

            else:
                self.resets[b] = -1
                self.dones[b] = False
                self.rewards[b] = -self._config.NPNP_SLACK_PENALTY
                self.infos[b] = {
                    "success": 0.0,
                    "failure": 0.0,
                    "pick_success": float(self._past_pick_success[b]),
                    "episode_steps": state.episode_step_idx,
                    "distance_to_start": ee_to_start,
                    "distance_to_goal": obj_to_goal,
                    "try_grasp": actions[(b * self.action_dim)]
                    > self._config.get("GRASP_THRESHOLD", 0.02),
                    "try_drop": actions[(b * self.action_dim)]
                    < self._config.get("DROP_THRESHOLD", 0.01),
                    "is_holding_correct": float(is_holding_correct),
                    "was_holding_correct": float(was_holding_correct),
                    "end_action": float(end_episode_action),
                    "_continuous_action_norm_mean": continuous_action_norm_mean,
                    "_continuous_action_norm_median": continuous_action_norm_median,
                    "_continuous_action_l2": continuous_action_l2,
                    "_original_drop_grasp": original_drop_grasp,
                    "_state.did_grasp": state.did_grasp,
                    "_state.did_attempt_grasp": state.did_attempt_grasp,
                    "_state.did_collide": state.did_collide,
                }
                if self._previous_state[b] is not None:
                    prev_obj_pos = prev_state.obj_positions[
                        prev_state.target_obj_idx
                    ]
                    curr_dist_ee_to_obj = (
                        state.obj_positions[state.target_obj_idx]
                        - state.ee_pos
                    ).length()
                    prev_dist_ee_to_obj = (
                        prev_obj_pos - prev_state.ee_pos
                    ).length()
                    self.rewards[b] += -(
                        curr_dist_ee_to_obj - prev_dist_ee_to_obj
                    ) * self._config.get("CARTHESIAN_REWARD", 1.0)
                    prev_obj_to_goal = (
                        prev_state.goal_pos - prev_obj_pos
                    ).length()
                    self.rewards[b] += -(
                        obj_to_goal - prev_obj_to_goal
                    ) * self._config.get("CARTHESIAN_REWARD", 1.0)
                    self.rewards[b] -= bad_attempt_penalty
                    self.rewards[b] -= action_penalty
                    if (
                        self._config.get("DROP_IS_FAIL", True)
                        and is_holding_correct
                        and (prev_state.held_obj_idx == -1)
                        and obj_to_goal
                        > self._config.NPNP_SUCCESS_THRESH  # the robot is not re-picking
                    ):
                        self.rewards[b] += self._config.PICK_REWARD
                self._previous_state[b] = state

    def reset(self):
        r"""Reset all the vectorized environments

        :return: list of outputs from the reset method of envs.
        """

        self.resets = np.full((self._num_envs), -1)
        self._current_episodes = [
            EnvironmentEpisodeState(-1) for _ in range(self._num_envs)
        ]

        for b in range(self._num_envs):
            next_episode = self.get_next_episode()
            if next_episode != -1:
                self.resets[b] = next_episode
                self._current_episodes[b].episode_id = next_episode
            else:
                # There are no more episodes to launch, so disable this env. We'll
                # hit this case during eval.
                # Note we don't yet communicate a disabled env to the sim; the
                # sim is assigned an arbitrary episode and we'll ignore the result.
                self.resets[b] = 0  # arbitrary episode
                self._current_episodes[b].set_disabled()

        if self._bsim:
            self._bsim.reset(self.resets)
            self._bsim.start_render()
            env_states = None  # self._bsim.get_environment_states()
            self.get_nonpixel_observations(env_states, self._observations)
            self._bsim.wait_render()
            self.get_pixel_observations(self._observations)

        self.rewards = np.zeros((self._num_envs), dtype=float)
        self.dones = np.zeros((self._num_envs), dtype=bool)
        self.resets = np.full((self._num_envs), -1)

        return self._observations

    @profiling_wrapper.RangeContext("async_step")
    def async_step(self, actions) -> None:
        r"""Asynchronously step in the environments."""
        scale = self._config.HACK_ACTION_SCALE
        if self._config.HACK_ACTION_SCALE != 1.0:
            actions = torch.mul(actions, scale)

        if self._bsim:
            assert self._config.OVERLAP_PHYSICS
            self._bsim.wait_step_physics_or_reset()
            self._bsim.start_render()
            env_states = None  # self._bsim.get_environment_states()
            self.get_nonpixel_observations(env_states, self._observations)
            if use_batch_dones_rewards_resets:
                modified_actions = self.get_batch_dones_rewards_resets(actions)
                self._bsim.start_step_physics_or_reset(
                    modified_actions.flatten().tolist(), self.resets
                )
            else:
                actions_flat_list = actions.flatten().tolist()
                assert (
                    len(actions_flat_list) == self.num_envs * self.action_dim
                )
                self.get_dones_rewards_resets(env_states, actions_flat_list)
                self._bsim.start_step_physics_or_reset(
                    actions_flat_list, self.resets
                )

    def fix_up_depth(self, raw_depth):
        obs = raw_depth.clamp(self.depth_sensor_config.MIN_DEPTH, self.depth_sensor_config.MAX_DEPTH)  # type: ignore[attr-defined]

        obs = obs.unsqueeze(-1)  # type: ignore[attr-defined]

        if self.depth_sensor_config.NORMALIZE_DEPTH:
            # normalize depth observation to [0, 1]
            obs = (obs - self.depth_sensor_config.MIN_DEPTH) / (
                self.depth_sensor_config.MAX_DEPTH
                - self.depth_sensor_config.MIN_DEPTH
            )

        return obs

    def get_pixel_observations(self, observations):
        if self._raw_depth is not None:
            observations["depth"] = self.fix_up_depth(self._raw_depth)

        if self._raw_rgb is not None:
            observations["rgb"] = self._raw_rgb

        if self._raw_debug_rgb is not None:
            observations["debug_rgb"] = self._raw_debug_rgb

    @profiling_wrapper.RangeContext("wait_step")
    def wait_step(
        self,
    ) -> Tuple[
        OrderedDict[str, Any], List[float], List[bool], List[Dict[str, Any]]
    ]:
        r"""Todo"""

        assert self._bsim

        # this updates self.raw_depth and self.raw_rgb
        # perf todo: ensure we're getting here before rendering finishes (issue a warning otherwise)
        self._bsim.wait_render()
        self.get_pixel_observations(self._observations)

        # these are "one frame behind" like the observations (i.e. computed from
        # the same earlier env state).
        rewards = self.rewards
        assert len(rewards) == self._num_envs
        dones = self.dones
        assert len(dones) == self._num_envs
        if self._config.REWARD_SCALE != 1.0:
            # perf todo: avoid dynamic list construction
            rewards = [r * self._config.REWARD_SCALE for r in rewards]

        # temp stub for infos
        # infos = [{"distance_to_goal": 0.0, "success":0.0, "spl":0.0}] * self._num_envs
        infos = self.infos
        return (self._observations, rewards, dones, infos)

    def step(
        self, actions
    ) -> Tuple[
        OrderedDict[str, Any], List[float], List[bool], List[Dict[str, Any]]
    ]:
        r"""Perform actions in the vectorized environments.

        :return: list of outputs from the step method of envs.
        """
        self.async_step(actions)
        return self.wait_step()

    def close(self) -> None:
        if self._is_closed:
            return

        self._bsim.close()
        self._bsim = None

        self._is_closed = True

    def pause_at(self, index: int) -> None:
        r"""Pauses computation on this env without destroying the env."""
        self._paused.append(index)

    def resume_all(self) -> None:
        r"""Resumes any paused envs."""
        self._paused = []

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

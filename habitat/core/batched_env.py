#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import magnum as mn
import numpy as np
from gym import spaces
from gym.spaces import Box

from habitat.tasks.utils import cartesian_to_polar
from habitat.utils import profiling_wrapper
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_sim.utils.common import quat_from_magnum

import torch  # isort:skip # noqa: F401  must import torch before importing bps_pytorch


class Camera:  # noqa: SIM119
    def __init__(self, attach_link_name, pos, rotation, hfov):
        self._attach_link_name = attach_link_name
        self._pos = pos
        self._rotation = rotation
        self._hfov = hfov


class StateSensorConfig:
    def __init__(self, config_key, shape, obs_key):
        self.config_key = config_key
        self.shape = shape
        self.obs_key = obs_key

    def get_obs(self, state):
        pass


def _get_spherical_coordinates(
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


class RobotStartSensorConfig(StateSensorConfig):
    def __init__(self):
        super().__init__("ROBOT_START_RELATIVE", 3, "robot_start_relative")

    def get_obs(self, state):
        robot_pos = state.robot_pos
        robot_rot = state.robot_rotation
        start_pos = state.robot_start_pos
        return _get_spherical_coordinates(robot_pos, start_pos, robot_rot)


class RobotTargetSensorConfig(StateSensorConfig):
    def __init__(self):
        super().__init__("ROBOT_TARGET_RELATIVE", 3, "robot_target_relative")

    def get_obs(self, state):
        robot_pos = state.robot_pos
        robot_rot = state.robot_rotation
        target_pos = state.target_obj_start_pos
        return _get_spherical_coordinates(robot_pos, target_pos, robot_rot)


class EEStartSensorConfig(StateSensorConfig):
    def __init__(self):
        super().__init__("EE_START_RELATIVE", 3, "ee_start_relative")

    def get_obs(self, state):
        ee_pos = state.ee_pos
        ee_rot = state.ee_rotation
        start_pos = state.robot_start_pos
        return _get_spherical_coordinates(ee_pos, start_pos, ee_rot)


class EETargetSensorConfig(StateSensorConfig):
    def __init__(self):
        super().__init__("EE_TARGET_RELATIVE", 3, "ee_target_relative")

    def get_obs(self, state):
        ee_pos = state.ee_pos
        ee_rot = state.ee_rotation
        target_pos = state.target_obj_start_pos
        return _get_spherical_coordinates(ee_pos, target_pos, ee_rot)


class RobotEESensorConfig(StateSensorConfig):
    def __init__(self):
        super().__init__("ROBOT_EE_RELATIVE", 3, "robot_ee_relative")

    def get_obs(self, state):
        robot_pos = state.robot_pos
        robot_rot = state.robot_rotation
        ee_pos = state.ee_pos
        return _get_spherical_coordinates(robot_pos, ee_pos, robot_rot)


class JointSensorConfig(StateSensorConfig):
    def __init__(self):
        super().__init__("JOINT_SENSOR", 7, "joint_pos")

    def get_obs(self, state):
        return state.robot_joint_positions[-9:-2]


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

        include_depth = "DEPTH_SENSOR" in config.SENSORS
        include_rgb = "RGB_SENSOR" in config.SENSORS

        self.state_sensor_config: List[StateSensorConfig] = []
        for ssc in [
            RobotStartSensorConfig(),
            RobotTargetSensorConfig(),
            EEStartSensorConfig(),
            EETargetSensorConfig(),
            JointSensorConfig(),
            RobotEESensorConfig(),
        ]:
            if ssc.config_key in config.SENSORS:
                self.state_sensor_config.append(ssc)

        assert include_depth or include_rgb

        self._num_envs = config.NUM_ENVIRONMENTS

        self._auto_reset_done = auto_reset_done
        self._config = config

        SIMULATOR_GPU_ID = self._config.SIMULATOR_GPU_ID
        agent_0_name = config.SIMULATOR.AGENTS[0]
        agent_0_config = getattr(config.SIMULATOR, agent_0_name)
        sensor_0_name = agent_0_config.SENSORS[0]
        agent_0_sensor_0_config = getattr(config.SIMULATOR, sensor_0_name)
        sensor_width, sensor_height = (
            agent_0_sensor_0_config.WIDTH,
            agent_0_sensor_0_config.HEIGHT,
        )

        if not config.STUB_BATCH_SIMULATOR:
            # require CUDA 11.0+ (lower versions will work but runtime perf will be bad!)
            assert torch.cuda.is_available() and torch.version.cuda.startswith(
                "11"
            )
            from habitat_sim._ext.habitat_sim_bindings import (
                BatchedSimulator,
                BatchedSimulatorConfig,
            )

            bsim_config = BatchedSimulatorConfig()
            bsim_config.gpu_id = SIMULATOR_GPU_ID
            bsim_config.include_depth = include_depth
            bsim_config.include_color = include_rgb
            bsim_config.num_envs = self._num_envs
            bsim_config.sensor0.width = sensor_width
            bsim_config.sensor0.height = sensor_height
            bsim_config.force_random_actions = False
            bsim_config.do_async_physics_step = self._config.OVERLAP_PHYSICS
            bsim_config.num_physics_substeps = (
                self._config.NUM_PHYSICS_SUBSTEPS
            )
            bsim_config.do_procedural_episode_set = True
            # bsim_config.episode_set_filepath = "../data/episode_sets/train.episode_set.json"
            self._bsim = BatchedSimulator(bsim_config)

            self.action_dim = self._bsim.get_num_actions()

            self._main_camera = Camera(
                "torso_lift_link",
                mn.Vector3(-0.536559, 1.16173, 0.568379),
                mn.Quaternion(
                    mn.Vector3(-0.26714, -0.541109, -0.186449), 0.775289
                ),
                60,
            )

            # reference code for wide-angle camera
            # self._main_camera = Camera("torso_lift_link",
            #     mn.Vector3(-0.536559, 2.16173, 0.568379),
            #     mn.Quaternion(mn.Vector3(-0.26714, -0.541109, -0.186449), 0.775289),
            #     75)

            self.set_camera(self._main_camera)

        else:
            self._bsim = None
            self.action_dim = 10  # arbitrary

        double_buffered = False
        buffer_index = 0

        observations = OrderedDict()
        if self._bsim:
            import bps_pytorch  # see https://github.com/shacklettbp/bps-nav#building

            if include_rgb:
                observations["rgb"] = bps_pytorch.make_color_tensor(
                    self._bsim.rgba(buffer_index),
                    SIMULATOR_GPU_ID,
                    self._num_envs // (2 if double_buffered else 1),
                    [sensor_height, sensor_width],
                )[..., 0:3].permute(
                    0, 1, 2, 3
                )  # todo: get rid of no-op permute

            if include_depth:
                observations["depth"] = bps_pytorch.make_depth_tensor(
                    self._bsim.depth(buffer_index),
                    SIMULATOR_GPU_ID,
                    self._num_envs // (2 if double_buffered else 1),
                    [sensor_height, sensor_width],
                ).unsqueeze(3)

        else:
            observations["rgb"] = (
                torch.rand(
                    [self._num_envs, sensor_height, sensor_width, 3],
                    dtype=torch.float32,
                )
                * 255
            )
            observations["depth"] = (
                torch.rand(
                    [self._num_envs, sensor_height, sensor_width, 1],
                    dtype=torch.float32,
                )
                * 255
            )
        for ssc in self.state_sensor_config:
            observations[ssc.obs_key] = torch.empty(
                [self._num_envs, ssc.shape], dtype=torch.float32
            )

        self._observations = observations

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
            depth_obs = spaces.Box(
                low=0.0,
                high=20.0,  # todo: investigate depth min/max
                shape=(
                    agent_0_sensor_0_config.HEIGHT,
                    agent_0_sensor_0_config.WIDTH,
                    1,
                ),
                dtype=np.float32,
            )
            obs_dict["depth"] = depth_obs

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

        self.rewards = [0.0] * self._num_envs
        self.dones = [False] * self._num_envs
        self.infos: List[Dict[str, Any]] = [{}] * self._num_envs
        self._previous_state: List[Optional[Any]] = [None] * self._num_envs
        self._previous_target_position = [None] * self._num_envs

        self.action_spaces = [
            action_space
        ] * 1  # note we only ever read element #0 of this array
        # self.number_of_episodes = []
        self._paused: List[int] = []
        self._num_episodes = self._bsim.get_num_episodes()
        self._next_episode_idx = 0

    def set_camera(self, camera):
        self._bsim.set_robot_camera(
            camera._attach_link_name,
            camera._pos,
            camera._rotation,
            camera._hfov,
        )

    @property
    def num_envs(self):
        r"""number of individual environments."""
        return self._num_envs - len(self._paused)

    def get_next_episode(self):
        assert self._num_episodes > 0
        retval = self._next_episode_idx
        self._next_episode_idx = (
            self._next_episode_idx + 1
        ) % self._num_episodes
        return retval

    def current_episodes(self):
        # todo: get current episode name from envs
        raise NotImplementedError()

    def count_episodes(self):
        raise NotImplementedError()

    def episode_over(self):
        raise NotImplementedError()

    def get_metrics(self):
        raise NotImplementedError()

    def get_nonpixel_observations(self, env_states, observations):
        for (b, state) in enumerate(env_states):
            for ssc in self.state_sensor_config:
                sensor_data = torch.tensor(ssc.get_obs(state))
                observations[ssc.obs_key][b, :] = sensor_data

    def get_dones_rewards_resets(self, env_states, actions):
        for (b, state) in enumerate(env_states):
            max_episode_len = 500

            # Target position is arbitrarily fixed
            local_target_position = mn.Vector3(0.6, 1, 0.6)

            global_target_position = quaternion_rotate_vector(
                quat_from_magnum(state.robot_rotation), local_target_position
            )
            global_target_position = state.robot_pos + mn.Vector3(
                global_target_position[0],
                global_target_position[1],
                global_target_position[2],
            )

            curr_dist = (global_target_position - state.ee_pos).length()
            success_dist = 0.05
            success = curr_dist < success_dist
            if success or state.episode_step_idx >= max_episode_len:
                self.dones[b] = True
                self.resets[b] = self.get_next_episode()
                self.rewards[b] = 10.0 if success else 0.0
                self.infos[b] = {
                    "success": float(success),
                    "episode_steps": state.episode_step_idx,
                }
                self._previous_state[b] = None
            else:
                self.resets[b] = -1
                self.dones[b] = False
                self.rewards[b] = 0
                self.infos[b] = {
                    "success": 0.0,
                    "episode_steps": state.episode_step_idx,
                }
                if self._previous_state[b] is not None:
                    last_dist = (
                        self._previous_target_position[b]
                        - self._previous_state[b].ee_pos
                    ).length()
                    self.rewards[b] = -(curr_dist - last_dist)
                self._previous_state[b] = state
                self._previous_target_position[b] = global_target_position

    def reset(self):
        r"""Reset all the vectorized environments

        :return: list of outputs from the reset method of envs.
        """

        self.resets = [self.get_next_episode() for _ in range(self._num_envs)]

        if self._bsim:
            self._bsim.reset(self.resets)
            self._bsim.start_render()
            env_states = self._bsim.get_environment_states()
            self.get_nonpixel_observations(env_states, self._observations)
            self._bsim.wait_render()

        self.rewards = [0.0] * self._num_envs
        self.dones = [False] * self._num_envs
        self.resets = [-1] * self._num_envs

        return self._observations

    def async_step(self, actions) -> None:
        r"""Asynchronously step in the environments."""
        scale = self._config.HACK_ACTION_SCALE
        if self._config.HACK_ACTION_SCALE != 1.0:
            actions = torch.mul(actions, scale)

        actions_flat_list = actions.flatten().tolist()
        assert len(actions_flat_list) == self.num_envs * self.action_dim
        if self._bsim:
            if self._config.OVERLAP_PHYSICS:
                self._bsim.wait_step_physics_or_reset()
                self._bsim.start_render()
                env_states = self._bsim.get_environment_states()

                self.get_nonpixel_observations(env_states, self._observations)
                self.get_dones_rewards_resets(env_states, actions_flat_list)
                self._bsim.start_step_physics_or_reset(
                    actions_flat_list, self.resets
                )

            else:
                # note: this path is untested
                self._bsim.start_step_physics_or_reset(
                    actions_flat_list, self.resets
                )
                self._bsim.wait_step_physics_or_reset()
                self._bsim.start_render()
                env_states = self._bsim.get_environment_states()
                self.get_dones_rewards_resets(env_states, actions_flat_list)

    @profiling_wrapper.RangeContext("wait_step")
    def wait_step(
        self,
    ) -> Tuple[
        OrderedDict[str, Any], List[float], List[bool], List[Dict[str, Any]]
    ]:
        r"""Todo"""

        if self._bsim:

            # this updates self._observations["depth"] (and rgb) tensors
            # perf todo: ensure we're getting here before rendering finishes (issue a warning otherwise)
            self._bsim.wait_render()

            # these are "one frame behind" like the observations (i.e. computed from
            # the same earlier env state).
            rewards = self.rewards
            assert len(rewards) == self._num_envs
            dones = self.dones
            assert len(dones) == self._num_envs
            if self._config.REWARD_SCALE != 1.0:
                # perf todo: avoid dynamic list construction
                rewards = [r * self._config.REWARD_SCALE for r in rewards]

        else:
            # rgb_observations = self._observations["rgb"]
            # torch.rand(rgb_observations.shape, dtype=torch.float32, out=rgb_observations)
            # torch.mul(rgb_observations, 255, out=rgb_observations)
            rewards = [0.0] * self._num_envs
            dones = [False] * self._num_envs
            infos: List[Dict[str, float]] = [{}] * self._num_envs

        observations = self._observations

        # temp stub for infos
        # infos = [{"distance_to_goal": 0.0, "success":0.0, "spl":0.0}] * self._num_envs
        infos = self.infos
        return (observations, rewards, dones, infos)

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

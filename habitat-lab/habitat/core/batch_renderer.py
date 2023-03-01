#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import OrderedDict
from typing import List

from omegaconf import DictConfig

import habitat_sim.errors
from habitat.core.registry import registry
from habitat.core.simulator import SensorSuite as CoreSensorSuite
from habitat.sims.habitat_simulator.habitat_simulator import (
    HabitatSimSensor,
    overwrite_config,
)
from habitat_sim import ReplayRenderer, ReplayRendererConfiguration


class BatchRenderer:
    r"""
    Wrapper for batch rendering functionality, which renders visual sensors of N environments simultaneously.

    The batch renderer pre-loads graphics data referenced by all simulators.

    When batch rendering, simulators add their state via an observation ("render_state").
    The batch renderer aggregates these observations and renders them all at once.
    """

    _enabled: bool = False
    _num_envs: int = 1
    _gpu_gpu: bool = False

    _sensor_suite: CoreSensorSuite = None
    _sensor_specifications: list = None

    _replay_renderer_cfg: ReplayRendererConfiguration = None
    _replay_renderer: ReplayRenderer = None

    def __init__(self, config: DictConfig, num_envs: int) -> None:
        r"""
        Initialize the batch renderer.

        param config: Base configuration.
        param num_envs: Number of concurrent environments to render.
        """
        self._enabled = config.habitat.simulator.enable_batch_renderer
        if not self._enabled:
            return

        self._num_envs = num_envs
        self._gpu_gpu = config.habitat.simulator.habitat_sim_v0.gpu_gpu

        self._sensor_suite = BatchRenderer._create_visual_core_sensor_suite(
            config
        )
        self._sensor_specifications = (
            BatchRenderer._create_sensor_specifications(
                config, self._sensor_suite
            )
        )
        self._replay_renderer_cfg = BatchRenderer._create_replay_renderer_cfg(
            config,
            self._num_envs,
            self._sensor_specifications,
        )
        self._replay_renderer: ReplayRenderer = (
            ReplayRenderer.create_batch_replay_renderer(
                self._replay_renderer_cfg
            )
        )

        # Pre-load dataset using composite GLTF file.
        if os.path.isfile(config.habitat.dataset.composite_file):
            print(
                "Pre-loading composite file: "
                + config.habitat.dataset.composite_file
            )
            self._replay_renderer.preload_file(
                config.habitat.dataset.composite_file
            )
        else:
            print(
                "No composite file pre-loaded. Batch rendering performance won't be optimal."
            )

    def render(self, observations: List[OrderedDict]) -> None:
        r"""
        Renders observations for all environments.
        This consumes "render_state" observations and adds results the observations.

        param observations: List of observations for each environment.
        """

    @staticmethod
    def _create_visual_core_sensor_suite(
        config: DictConfig,
    ) -> CoreSensorSuite:
        r"""
        Instantiates a core sensor suite from configuration that only contains visual sensors.
        """
        sim_sensors = []
        for agent_cfg in config.habitat.simulator.agents.values():
            for sensor_cfg in agent_cfg.sim_sensors.values():
                sensor_type = registry.get_sensor(sensor_cfg.type)
                if sensor_type.sim_sensor_type in [  # type: ignore
                    habitat_sim.SensorType.COLOR,
                    habitat_sim.SensorType.DEPTH,
                ]:
                    sim_sensors.append(sensor_type(sensor_cfg))
        return CoreSensorSuite(sim_sensors)

    @staticmethod
    def _create_sensor_specifications(
        config: DictConfig, sensor_suite: CoreSensorSuite
    ) -> list:
        r"""
        Creates a list of Habitat-Sim sensor specifications from a specified core sensor suite.
        """
        sensor_specifications: list = []
        for sensor in sensor_suite.sensors.values():
            assert isinstance(sensor, HabitatSimSensor)
            sim_sensor_cfg = sensor._get_default_spec()  # type: ignore
            overwrite_config(
                config_from=sensor.config,
                config_to=sim_sensor_cfg,
                # These keys are only used by Hab-Lab
                # or translated into the sensor config manually
                ignore_keys=sensor._config_ignore_keys,
                # TODO consider making trans_dict a sensor class var too.
                trans_dict={
                    "sensor_model_type": lambda v: getattr(
                        habitat_sim.FisheyeSensorModelType, v
                    ),
                    "sensor_subtype": lambda v: getattr(
                        habitat_sim.SensorSubType, v
                    ),
                },
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
            sim_sensor_cfg.gpu2gpu_transfer = (
                config.habitat.simulator.habitat_sim_v0.gpu_gpu
            )
            sensor_specifications.append(sim_sensor_cfg)
        return sensor_specifications

    @staticmethod
    def _create_replay_renderer_cfg(
        config: DictConfig, num_env: int, sensor_specifications: list
    ) -> ReplayRendererConfiguration:
        r"""
        Creates the configuration info for creating a replay renderer.
        """
        replay_renderer_cfg: ReplayRendererConfiguration = (
            ReplayRendererConfiguration()
        )
        replay_renderer_cfg.num_environments = num_env
        replay_renderer_cfg.standalone = True
        replay_renderer_cfg.sensor_specifications = sensor_specifications
        replay_renderer_cfg.gpu_device_id = (
            config.habitat.simulator.habitat_sim_v0.gpu_device_id
        )
        replay_renderer_cfg.force_separate_semantic_scene_graph = False
        replay_renderer_cfg.leave_context_with_background_renderer = False
        return replay_renderer_cfg

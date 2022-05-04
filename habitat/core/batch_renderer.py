#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import habitat_sim.errors
from habitat.core.registry import registry
from habitat.core.simulator import SensorSuite as CoreSensorSuite
from habitat.sims.habitat_simulator.habitat_simulator import (
    HabitatSimSensor,
    overwrite_config,
)
from habitat_sim import ReplayBatchRenderer, ReplayBatchRendererConfiguration
from habitat_sim.sensors.sensor_suite import SensorSuite as BackendSensorSuite
from habitat_sim.simulator import Sensor as BackendSensor


# need an object with this API for Sensor
class DummyAgent:
    def __init__(self):
        self.scene_node = (
            None  # not scene root node; just a dummy node to parent sensor to
        )
        self._sensors = (
            BackendSensorSuite()
        )  # collection of c++ CameraSensor objects created by hsim.SensorFactory.create_sensors


class EnvironmentRecord:
    def __init__(self):
        # todo: consider dummy sim; also consider revising CoreSensor class to not use "agent", "sim"
        self.dummy_agent = DummyAgent()
        self._sensors = {}  # dict of Python Sensors by uuid


# todo: incorporate post_step into envs interface; create VectorEnv with BatchRenderer
# but beware slicing stuff
class BatchRenderer:
    def get_habitat_config(self):
        return self.config.TASK_CONFIG.SIMULATOR

    def __init__(self, config):
        self.config = config

        self.env_records = [
            EnvironmentRecord() for _ in range(self.config.NUM_ENVIRONMENTS)
        ]

        self.init_shared_sensors()

        self.gpu_device = (
            0  # needed for Sensor API; todo: get from config yaml
        )
        self.frustum_culling = True  # needed for Sensor API

        self.backend_config = ReplayBatchRendererConfiguration()
        self.backend_config.num_environments = self.config.NUM_ENVIRONMENTS
        self.backend_config.gpu_device_id = self.gpu_device
        self.backend_config.sensor_specifications = self.sensor_specifications

        self.sensor_user_prefix = "sensor_"

        self.backend = ReplayBatchRenderer(self.backend_config)

        self.renderer = self.backend.renderer

        self.init_environment_sensors()

        self._async_draw_agent_ids = None
        self.active_env_index = None

    def init_shared_sensors(self):

        # todo: only visual sensors

        sim_sensors = []
        for sensor_name in self.config.SENSORS:
            sensor_cfg = getattr(self.get_habitat_config(), sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = CoreSensorSuite(sim_sensors)

        sensor_specifications = []

        for sensor in self._sensor_suite.sensors.values():
            assert isinstance(sensor, HabitatSimSensor)
            sim_sensor_cfg = sensor._get_default_spec()  # type: ignore[misc]
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

            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            # We know that the Sensor has to be one of these Sensors
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.get_habitat_config().HABITAT_SIM_V0.GPU_GPU
            )
            sensor_specifications.append(sim_sensor_cfg)

        self.sensor_specifications = sensor_specifications

    def init_environment_sensors(self):

        for env_index in range(self.backend_config.num_environments):
            env_record = self.env_records[env_index]

            env_record.dummy_agent.scene_node = (
                self.backend.get_environment_sensor_parent_node(env_index)
            )

            sensor_suite = self.backend.get_environment_sensors(
                env_index
            )  # todo: only call this once; cache result

            for spec in self.sensor_specifications:

                env_record.dummy_agent._sensors.add(sensor_suite[spec.uuid])
                # this (BatchRenderer) masquerading as Sim
                env_record._sensors[spec.uuid] = BackendSensor(
                    sim=self, agent=env_record.dummy_agent, sensor_id=spec.uuid
                )

    def __set_from_config(self, sensor_specifications) -> None:
        # self._config_backend(config)
        # self._config_agents(config)
        # self._config_pathfinder(config)
        self.frustum_culling = True  # config.sim_cfg.frustum_culling

    # Sensor needs this API
    def get_active_semantic_scene_graph(self):
        assert self.active_env_index is not None
        return self.backend.get_semantic_scene_graph(self.active_env_index)

    # Sensor needs this API
    def get_active_scene_graph(self):
        assert self.active_env_index is not None
        return self.backend.get_scene_graph(self.active_env_index)

    # todo
    # def close(self, destroy: bool = True) -> None:
    #     r"""Close the simulator instance.

    #     :param destroy: Whether or not to force the OpenGL context to be
    #         destroyed if async rendering was used.  If async rendering wasn't used,
    #         this has no effect.
    #     """
    #     # NB: Python still still call __del__ (and thus)
    #     # close even if __init__ errors. We don't
    #     # have anything to close if we aren't initialized so
    #     # we can just return.
    #     if not self._initialized:
    #         return

    #     if self.renderer is not None:
    #         self.renderer.acquire_gl_context()

    #     self.backend.close()

    def post_step(self, observations):

        assert len(observations) == self.backend_config.num_environments
        for env_index in range(self.backend_config.num_environments):
            observations_for_env = observations[env_index]
            # grab and remove sim_blob
            sim_blob = observations_for_env.pop("sim_blob")
            self.backend.set_environment_keyframe(env_index, sim_blob)
            self.backend.set_sensor_transforms_from_keyframe(
                env_index, self.sensor_user_prefix
            )

        self.start_async_render()
        # todo: make similar to rearrange_sim.py:
        # self._prev_sim_obs = self.get_sensor_observations_async_finish()
        # obs = self._sensor_suite.get_observations(self._prev_sim_obs)
        self.get_sensor_observations_async_finish(observations)
        return observations

    def start_async_render(self):
        if self._async_draw_agent_ids is not None:
            raise RuntimeError(
                "start_async_render_and_step_physics was already called.  "
                "Call get_sensor_observations_async_finish before calling this again.  "
                "Use step_physics to step physics additional times."
            )

        agent_ids = [0]
        self._async_draw_agent_ids = agent_ids

        assert self.active_env_index is None
        for env_index in range(self.backend_config.num_environments):
            # __sensors = self.backend.get_environment_sensors(env_index)  # todo: only call this once; cache result
            # sensorsuite = __sensors
            # for _sensor_uuid, sensor in sensorsuite.items():
            #     sensor._draw_observation_async()
            self.active_env_index = env_index
            env_record = self.env_records[env_index]
            for _, sensor in env_record._sensors.items():
                sensor._draw_observation_async()
        self.active_env_index = None

        self.renderer.start_draw_jobs()

    def get_sensor_observations_async_finish(self, observations_by_env):
        if self._async_draw_agent_ids is None:
            raise RuntimeError(
                "get_sensor_observations_async_finish was called before calling start_async_render_and_step_physics."
            )

        # todo: get rid of _async_draw_agent_ids stuff
        agent_ids = self._async_draw_agent_ids
        self._async_draw_agent_ids = None
        if isinstance(agent_ids, int):
            agent_ids = [agent_ids]

        self.renderer.wait_draw_jobs()

        assert self.active_env_index is None
        for env_index, observations in enumerate(observations_by_env):

            self.active_env_index = env_index
            env_record = self.env_records[env_index]
            raw_visual_observations = dict()
            for sensor_uuid, sensor in env_record._sensors.items():
                raw_visual_observations[
                    sensor_uuid
                ] = sensor._get_observation_async()

            # is it okay to re-use _sensor_suite across multiple envs?
            processed_visual_observations = (
                self._sensor_suite.get_observations(raw_visual_observations)
            )
            # todo: check this logic
            observations.update(processed_visual_observations)
        self.active_env_index = None

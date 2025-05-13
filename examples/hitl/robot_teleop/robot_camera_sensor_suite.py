#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Optional

import magnum as mn

import habitat_sim
from habitat_hitl.core.hydra_utils import omegaconf_to_object


@dataclass
class SensorCacheEntry:
    publish_topic: Optional[str] = None
    sim_sensor: Optional[Any] = None
    recent_obs = None


class RobotCameraSensorSuite:
    def __init__(self, sim, robot_ao, camera_sensors_config):
        self.sim = sim
        if self.sim.renderer is None:
            raise RuntimeError(
                "RobotCameraSensorSuite requires a sim with a renderer. See hitl_defaults.yaml enable_sim_driver_renderer."
            )
        self._robot_ao = robot_ao
        self._configs = omegaconf_to_object(camera_sensors_config)
        # quick sanity check that we have the right kind of camera_sensors config
        assert (
            isinstance(self._configs, list)
            and len(self._configs)
            and hasattr(self._configs[0], "sensor_uuid")
        )
        self._equirect = False
        self.clear_color = mn.Color4.from_linear_rgb_int(0)
        self.agent: habitat_sim.simulator.Agent = None
        self.agent_id = 0

        self.sensor_cache = {}
        for config in self._configs:
            self.sensor_cache[config.sensor_uuid] = SensorCacheEntry()

        self._create_agent_and_sensors()

        from murp.mock.mock_camera_suite_topics import MockCameraSuiteTopics

        self.topics_by_sensor = {}
        for config in self._configs:
            topic = MockCameraSuiteTopics.find_full_topic(config.sensor_uuid)
            self.sensor_cache[config.sensor_uuid].publish_topic = topic

    def close(self):
        # NOTE: this guards against cases where the Simulator is deconstructed before the DBV
        if self.agent_id < len(self.sim.agents):
            # remove the agent and sensor from the Simulator instance
            self.agent.close()
            del self.sim._Simulator__sensors[self.agent_id]
            del self.sim.agents[self.agent_id]

        self.agent = None
        self.agent_id = 0
        self.sensors = None

    def _create_sensor_spec_from_config(self, config):
        debug_sensor_spec = (
            habitat_sim.CameraSensorSpec()
            if not self._equirect
            else habitat_sim.EquirectangularSensorSpec()
        )
        debug_sensor_spec.sensor_type = (
            habitat_sim.SensorType.COLOR
            if config.sensor_type == "color"
            else habitat_sim.SensorType.DEPTH
        )
        debug_sensor_spec.position = [0.0, 0.0, 0.0]
        debug_sensor_spec.resolution = [
            config.resolution[0],
            config.resolution[1],
        ]
        debug_sensor_spec.uuid = config.sensor_uuid
        debug_sensor_spec.clear_color = self.clear_color
        debug_sensor_spec.hfov = config.hfov

        return debug_sensor_spec

    def _create_agent_and_sensors(self):
        sensor_specifications = []
        for config in self._configs:
            sensor_specifications.append(
                self._create_sensor_spec_from_config(config)
            )

        debug_agent_config = habitat_sim.agent.AgentConfiguration()
        debug_agent_config.sensor_specifications = sensor_specifications
        self.sim.agents.append(
            habitat_sim.Agent(
                self.sim.get_active_scene_graph()
                .get_root_node()
                .create_child(),
                debug_agent_config,
            )
        )
        self.agent = self.sim.agents[-1]
        self.agent_id = len(self.sim.agents) - 1
        self.sim._Simulator__sensors.append({})
        self.sensors = {}
        for config in self._configs:
            self.sim._update_simulator_sensors(
                config.sensor_uuid, self.agent_id
            )
            self.sensor_cache[
                config.sensor_uuid
            ].sim_sensor = self.sim._Simulator__sensors[self.agent_id][
                config.sensor_uuid
            ]

    def _update_sensor_transforms(self):
        # this should be identity
        inv_T = self.agent.scene_node.transformation.inverted()

        for config in self._configs:
            link_trans = self._robot_ao.get_link_scene_node(
                config.attached_link_id
            ).transformation

            pos = mn.Vector3(config.cam_offset_pos)
            ori = mn.Vector3(config.cam_orientation)
            Mt = mn.Matrix4.translation(pos)
            Mz = mn.Matrix4.rotation_z(mn.Rad(ori[2]))
            My = mn.Matrix4.rotation_y(mn.Rad(ori[1]))
            Mx = mn.Matrix4.rotation_x(mn.Rad(ori[0]))
            cam_transform = Mt @ Mz @ My @ Mx

            cam_info_relative_transform = mn.Matrix4.rotation_z(mn.Deg(-90))

            cam_transform = (
                link_trans @ cam_transform @ cam_info_relative_transform
            )
            # todo: assert inv_T is identity and then remove this line
            cam_transform = inv_T @ cam_transform

            sim_sensor = self.sensor_cache[config.sensor_uuid].sim_sensor
            # sim_sensor._sensor_object.node

            from habitat_sim.utils.common import orthonormalize_rotation_shear

            sim_sensor._sensor_object.node.transformation = (
                orthonormalize_rotation_shear(cam_transform)
            )

    def draw_debug(self, gui_drawer):
        for config in self._configs:
            sim_sensor = self.sensor_cache[config.sensor_uuid].sim_sensor
            gui_drawer.draw_axes(
                sim_sensor._sensor_object.node.transformation, scale=0.5
            )

    # todo: rename to convey that this does drawing/rendering
    def _draw_observations(
        self,
    ):
        assert self.sensors is not None

        self._update_sensor_transforms()

        for config in self._configs:
            entry = self.sensor_cache[config.sensor_uuid]
            sensor = entry.sim_sensor
            sensor.draw_observation()
            entry.recent_obs = sensor.get_observation()

    def get_recent_observations(self):
        observations: list[Any] = []
        for config in self._configs:
            entry = self.sensor_cache[config.sensor_uuid]
            observations.append(entry.recent_obs)
        return observations

    def draw_and_publish_observations(self, mock_camera_suite):
        self._draw_observations()

        suite = mock_camera_suite
        for config in self._configs:
            entry = self.sensor_cache[config.sensor_uuid]
            suite.publish_image_rgb_or_depth(
                entry.publish_topic, entry.recent_obs
            )

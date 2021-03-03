#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from habitat.core.registry import registry
from habitat.core.simulator import Config
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_sim.agent import ActionSpec, ActuationSpec
from habitat_sim.nav import NavMeshSettings


@registry.register_simulator(name="RearrangementSim-v0")
class RearrangementSim(HabitatSim):
    r"""Simulator wrapper over habitat-sim with
    object rearrangement functionalities.
    """

    def __init__(self, config: Config) -> None:
        self.did_reset = False
        super().__init__(config=config)
        self.grip_offset = np.eye(4)

        self.agent_id = self.habitat_config.DEFAULT_AGENT_ID
        agent_config = self._get_agent_config(self.agent_id)

        self.navmesh_settings = NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_radius = agent_config.RADIUS
        self.navmesh_settings.agent_height = agent_config.HEIGHT

        self._prev_sim_obs = {}
        self.sim_object_to_objid_mapping = {}
        self.objid_to_sim_object_mapping = {}

    def reconfigure(self, config: Config) -> None:
        super().reconfigure(config)

    def _rotate_agent_sensors(self):
        r"""Rotates the sensor to look down at the start of the episode.
        Args:
            mode: sensor whose observation is used for returning the frame,
                eg: "rgb", "depth", "semantic"

        Returns:
            rendered frame according to the mode
        """
        agent = self.get_agent(self.agent_id)

        for _, v in agent._sensors.items():
            action = ActionSpec(
                name="look_down",
                actuation=ActuationSpec(
                    amount=self.habitat_config.INITIAL_LOOK_DOWN_ANGLE
                ),
            )

            agent.controls.action(
                v.object, action.name, action.actuation, apply_filter=False
            )

    def reset(self):
        sim_obs = super().reset()

        if self._update_agents_state():
            self._rotate_agent_sensors()
            sim_obs = self.get_sensor_observations()

        self._prev_sim_obs.update(sim_obs)
        self._prev_sim_obs["gripped_object_id"] = -1
        self._prev_sim_obs["collided"] = False
        self.did_reset = True
        self.grip_offset = np.eye(4)
        return self._sensor_suite.get_observations(sim_obs)

    def _sync_gripped_object(self, gripped_object_id):
        r"""
        Sync the gripped object with the object associated with the agent.
        """
        if gripped_object_id != -1:
            agent_body_transformation = (
                self._default_agent.scene_node.transformation
            )
            self.set_transformation(
                agent_body_transformation, gripped_object_id
            )
            translation = agent_body_transformation.transform_point(
                np.array([0, 2.0, 0])
            )
            self.set_translation(translation, gripped_object_id)

    @property
    def gripped_object_id(self):
        return self._prev_sim_obs.get("gripped_object_id", -1)

    def step(self, action: int):
        dt = 1 / 60.0
        self._num_total_frames += 1
        collided = self._default_agent.act(action)
        self._last_state = self._default_agent.get_state()

        # step physics by dt
        super().step_world(dt)

        # Sync the gripped object after the agent moves.
        self._sync_gripped_object(self._prev_sim_obs["gripped_object_id"])

        # obtain observations
        self._prev_sim_obs.update(self.get_sensor_observations())
        self._prev_sim_obs["collided"] = collided

        observations = self._sensor_suite.get_observations(self._prev_sim_obs)
        return observations

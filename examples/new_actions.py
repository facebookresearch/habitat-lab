#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


r"""
This is an example of how to add new actions to habitat-lab


We will use the strafe action outline in the habitat_sim example
"""

import attr
import numpy as np

import habitat
import habitat_sim
from habitat.config.default_structured_configs import ActionConfig
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat.tasks.nav.nav import SimulatorTaskAction


@attr.s(auto_attribs=True, slots=True)
class NoisyStrafeActuationSpec:
    move_amount: float
    # Classic strafing is to move perpendicular (90 deg) to the forward direction
    strafe_angle: float = 90.0
    noise_amount: float = 0.05


def _strafe_impl(
    scene_node: habitat_sim.SceneNode,
    move_amount: float,
    strafe_angle: float,
    noise_amount: float,
):
    forward_ax = (
        np.array(scene_node.absolute_transformation().rotation_scaling())
        @ habitat_sim.geo.FRONT
    )
    strafe_angle = np.deg2rad(strafe_angle)
    strafe_angle = np.random.uniform(
        (1 - noise_amount) * strafe_angle, (1 + noise_amount) * strafe_angle
    )

    rotation = habitat_sim.utils.quat_from_angle_axis(
        strafe_angle, habitat_sim.geo.UP
    )
    move_ax = habitat_sim.utils.quat_rotate_vector(rotation, forward_ax)

    move_amount = np.random.uniform(
        (1 - noise_amount) * move_amount, (1 + noise_amount) * move_amount
    )
    scene_node.translate_local(move_ax * move_amount)


@habitat_sim.registry.register_move_fn(body_action=True)
class NoisyStrafeLeft(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: NoisyStrafeActuationSpec,
    ):
        print(f"strafing left with noise_amount={actuation_spec.noise_amount}")
        _strafe_impl(
            scene_node,
            actuation_spec.move_amount,
            actuation_spec.strafe_angle,
            actuation_spec.noise_amount,
        )


@habitat_sim.registry.register_move_fn(body_action=True)
class NoisyStrafeRight(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: NoisyStrafeActuationSpec,
    ):
        print(
            f"strafing right with noise_amount={actuation_spec.noise_amount}"
        )
        _strafe_impl(
            scene_node,
            actuation_spec.move_amount,
            -actuation_spec.strafe_angle,
            actuation_spec.noise_amount,
        )


@habitat.registry.register_action_space_configuration
class NoNoiseStrafe(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.STRAFE_LEFT] = habitat_sim.ActionSpec(
            "noisy_strafe_left",
            NoisyStrafeActuationSpec(0.25, noise_amount=0.0),
        )
        config[HabitatSimActions.STRAFE_RIGHT] = habitat_sim.ActionSpec(
            "noisy_strafe_right",
            NoisyStrafeActuationSpec(0.25, noise_amount=0.0),
        )

        return config


@habitat.registry.register_action_space_configuration
class NoiseStrafe(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.STRAFE_LEFT] = habitat_sim.ActionSpec(
            "noisy_strafe_left",
            NoisyStrafeActuationSpec(0.25, noise_amount=0.05),
        )
        config[HabitatSimActions.STRAFE_RIGHT] = habitat_sim.ActionSpec(
            "noisy_strafe_right",
            NoisyStrafeActuationSpec(0.25, noise_amount=0.05),
        )

        return config


@habitat.registry.register_task_action
class StrafeLeft(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "strafe_left"

    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.STRAFE_LEFT)


@habitat.registry.register_task_action
class StrafeRight(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "strafe_right"

    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.STRAFE_RIGHT)


def main():
    HabitatSimActions.extend_action_space("STRAFE_LEFT")
    HabitatSimActions.extend_action_space("STRAFE_RIGHT")

    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )
    with habitat.config.read_write(config):
        # Add a simple action config to the config.habitat.task.actions dictionary
        config.habitat.task.actions["STRAFE_LEFT"] = ActionConfig(
            type="StrafeLeft"
        )
        config.habitat.task.actions["STRAFE_RIGHT"] = ActionConfig(
            type="StrafeRight"
        )

        config.habitat.simulator.action_space_config = "NoNoiseStrafe"

    with habitat.Env(config=config) as env:
        env.reset()
        env.step("STRAFE_LEFT")
        env.step("STRAFE_RIGHT")

    with habitat.config.read_write(config):
        config.habitat.simulator.action_space_config = "NoiseStrafe"

    with habitat.Env(config=config) as env:
        env.reset()
        env.step("STRAFE_LEFT")
        env.step("STRAFE_RIGHT")


if __name__ == "__main__":
    main()

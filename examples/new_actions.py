#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


r"""
This is an example of how to add new actions to habitat-lab


We will use the strafe action outline in the habitat_sim example
"""

from dataclasses import dataclass

import magnum as mn
import numpy as np

import habitat
from habitat.config.default_structured_configs import ActionConfig
from habitat.tasks.nav.nav import SimulatorTaskAction


# This is the configuration for our action.
@dataclass
class StrafeActionConfig(ActionConfig):
    move_amount: float = 0.0  # We will change this in the configuration
    noise_amount: float = 0.0


# This is a helper that implements strafing that we will use in our actions
def _strafe_body(
    sim,
    move_amount: float,
    strafe_angle_deg: float,
    noise_amount: float,
):
    # Get the state of the agent
    agent_state = sim.get_agent_state()
    # Convert from np.quaternion (quaternion.quaternion) to mn.Quaternion
    normalized_quaternion = agent_state.rotation
    agent_mn_quat = mn.Quaternion(
        normalized_quaternion.imag, normalized_quaternion.real
    )
    forward = agent_mn_quat.transform_vector(-mn.Vector3.z_axis())
    strafe_angle = np.random.uniform(
        (1 - noise_amount) * strafe_angle_deg,
        (1 + noise_amount) * strafe_angle_deg,
    )
    strafe_angle = mn.Deg(strafe_angle)
    rotation = mn.Quaternion.rotation(strafe_angle, mn.Vector3.y_axis())
    move_amount = np.random.uniform(
        (1 - noise_amount) * move_amount, (1 + noise_amount) * move_amount
    )
    delta_position = rotation.transform_vector(forward) * move_amount
    final_position = sim.pathfinder.try_step(  # type: ignore
        agent_state.position, agent_state.position + delta_position
    )
    sim.set_agent_state(
        final_position,
        [*rotation.vector, rotation.scalar],
        reset_sensors=False,
    )


# We define and register our actions as follows.
# the __init__ method receives a sim and config argument.
@habitat.registry.register_task_action
class StrafeLeft(SimulatorTaskAction):
    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim = sim
        self._move_amount = config.move_amount
        self._noise_amount = config.noise_amount

    def _get_uuid(self, *args, **kwargs) -> str:
        return "strafe_left"

    def step(self, *args, **kwargs):
        print(
            f"Calling {self._get_uuid()} d={self._move_amount}m noise={self._noise_amount}"
        )
        # This is where the code for the new action goes. Here we use a
        # helper method but you could directly modify the simulation here.
        _strafe_body(self._sim, self._move_amount, 90, self._noise_amount)


@habitat.registry.register_task_action
class StrafeRight(SimulatorTaskAction):
    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim = sim
        self._move_amount = config.move_amount
        self._noise_amount = config.noise_amount

    def _get_uuid(self, *args, **kwargs) -> str:
        return "strafe_right"

    def step(self, *args, **kwargs):
        print(
            f"Calling {self._get_uuid()} d={self._move_amount}m noise={self._noise_amount}"
        )
        _strafe_body(self._sim, self._move_amount, -90, self._noise_amount)


def main():
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )
    with habitat.config.read_write(config):
        # Add a simple action config to the config.habitat.task.actions dictionary
        # Here we do it via code, but you can easily add them to a yaml config as well
        config.habitat.task.actions["STRAFE_LEFT"] = StrafeActionConfig(
            type="StrafeLeft",
            move_amount=0.25,
            noise_amount=0.0,
        )
        config.habitat.task.actions["STRAFE_RIGHT"] = StrafeActionConfig(
            type="StrafeRight",
            move_amount=0.25,
            noise_amount=0.0,
        )
        config.habitat.task.actions["NOISY_STRAFE_LEFT"] = StrafeActionConfig(
            type="StrafeLeft",
            move_amount=0.25,
            noise_amount=0.05,  # We add some noise to the configuration here
        )
        config.habitat.task.actions["NOISY_STRAFE_RIGHT"] = StrafeActionConfig(
            type="StrafeRight",
            move_amount=0.25,
            noise_amount=0.05,  # We add some noise to the configuration here
        )

    with habitat.Env(config=config) as env:
        env.reset()
        env.step("STRAFE_LEFT")
        env.step("STRAFE_RIGHT")
        env.step("NOISY_STRAFE_LEFT")
        env.step("NOISY_STRAFE_RIGHT")


if __name__ == "__main__":
    main()

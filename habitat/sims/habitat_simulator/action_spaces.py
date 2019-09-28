from enum import Enum

import attr

import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import (
    ActionSpaceConfiguration,
    Config,
    SimulatorActions,
)


@registry.register_action_space_configuration(name="v0")
class HabitatSimV0ActionSpaceConfiguration(ActionSpaceConfiguration):
    def get(self):
        return {
            SimulatorActions.STOP: habitat_sim.ActionSpec("stop"),
            SimulatorActions.MOVE_FORWARD: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            SimulatorActions.TURN_LEFT: habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            SimulatorActions.TURN_RIGHT: habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
        }


@registry.register_action_space_configuration(name="v1")
class HabitatSimV1ActionSpaceConfiguration(
    HabitatSimV0ActionSpaceConfiguration
):
    def get(self):
        config = super().get()
        new_config = {
            SimulatorActions.LOOK_UP: habitat_sim.ActionSpec(
                "look_up",
                habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
            SimulatorActions.LOOK_DOWN: habitat_sim.ActionSpec(
                "look_down",
                habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
        }

        config.update(new_config)

        return config

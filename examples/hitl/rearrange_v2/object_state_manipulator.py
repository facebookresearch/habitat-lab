#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

from habitat_llm.sims.collaboration_sim import CollaborationSim
from habitat_llm.tools.motor_skills.object_states.oracle_clean_skills import (
    OracleCleanInPlaceSkill,
)
from habitat_llm.tools.motor_skills.object_states.oracle_fill_skills import (
    OracleFillInPlaceSkill,
)
from habitat_llm.tools.motor_skills.object_states.oracle_object_state_skill import (
    OracleObjectStateInPlaceSkill,
)
from habitat_llm.tools.motor_skills.object_states.oracle_pour_skills import (
    OraclePourInPlaceSkill,
)
from habitat_llm.tools.motor_skills.object_states.oracle_power_skills import (
    OraclePowerOffInPlaceSkill,
    OraclePowerOnInPlaceSkill,
)
from world import World

from habitat.sims.habitat_simulator import sim_utilities
from habitat.sims.habitat_simulator.object_state_machine import (
    ObjectStateSpec,
    set_state_of_obj,
)

BooleanActionMap = Dict[bool, OracleObjectStateInPlaceSkill]


@dataclass
class ObjectStateChangeResult:
    success: bool
    error_message: Optional[str]


@dataclass
class BooleanAction:
    state_spec: ObjectStateSpec
    current_value: bool
    target_value: bool
    available: bool
    enabled: bool
    error: Optional[str]


class ObjectStateManipulator:
    """
    Helper class for reading and writing object states for a given world, on behalf of a given agent.
    Requires the 'habitat_llm' external module.
    """

    def __init__(
        self,
        sim: CollaborationSim,
        agent_index: int,
        world: World,
        maximum_distance: float,
    ):
        self._sim = sim
        self._agent_index = agent_index
        self._maximum_distance = maximum_distance
        self._world = world

        self._boolean_state_action_map: Dict[str, BooleanActionMap] = {
            "is_clean": {True: OracleCleanInPlaceSkill},
            "is_filled": {
                True: OracleFillInPlaceSkill,
                False: OraclePourInPlaceSkill,
            },
            "is_powered_on": {
                True: OraclePowerOnInPlaceSkill,
                False: OraclePowerOffInPlaceSkill,
            },
        }
        """
        Map of boolean state values and their associated skill.

        Example:
        ```
        "is_powered_on": {
            True: PowerOnSkill,
            False: PowerOffSkill,
        },
        ```
        """

    def set_object_state(
        self, object_handle: str, state_name: str, state_value: Any
    ) -> None:
        """
        Set an object state, regardless of constraints.
        """
        obj = sim_utilities.get_obj_from_handle(self._sim, object_handle)
        set_state_of_obj(obj, state_name, state_value)

    def get_boolean_action(
        self, state_name: str, target_value: bool
    ) -> Optional[OracleObjectStateInPlaceSkill]:
        """
        Return a boolean action that allows for changing 'state_name' to 'target_value'.
        Returns None if no action is available to do this state change.
        """
        actions = self._boolean_state_action_map
        if state_name not in actions:
            return None
        return actions[state_name].get(target_value, None)

    def get_action(
        self, state_name: str, target_value: Any
    ) -> Optional[OracleObjectStateInPlaceSkill]:
        """
        Return an action that allows for changing 'state_name' to 'target_value'.
        Returns None if no action is available to do this state change.
        """
        if type(target_value) == bool:
            return self.get_boolean_action(state_name, target_value)
        else:
            # Unsupported action type.
            return None

    def can_execute_action(
        self, state_name: str, state_value: Any, object_handle: str
    ) -> ObjectStateChangeResult:
        """
        Test whether an action can be executed on the specified object.
        Returns whether the action can be executed, and an error message if it can't.
        """
        action = self.get_action(state_name, state_value)
        if action is not None:
            result = action.can_modify_state_impl(
                self._sim,
                self._agent_index,
                object_handle,
                state_name,
                self._maximum_distance,
                self._world._metadata_interface,
            )
            return ObjectStateChangeResult(
                result.succeeded, result.error_message_user
            )
        return ObjectStateChangeResult(
            False, f"Undefined action: '{state_name} -> {state_value}'."
        )

    def try_execute_action(
        self, state_name: str, state_value: Any, object_handle: str
    ) -> ObjectStateChangeResult:
        """
        Try executing an action on the specified object.
        Returns whether the action was successful, and an error message if it wasn't.
        """
        result = self.can_execute_action(
            state_name, state_value, object_handle
        )
        if result.success:
            self.set_object_state(object_handle, state_name, state_value)
        return ObjectStateChangeResult(result.success, result.error_message)

    def get_all_available_boolean_actions(
        self, object_handle: str
    ) -> List[BooleanAction]:
        """
        Return all boolean actions available for the specified object.
        """
        actions: List[BooleanAction] = []
        world = self._world
        states = world.get_states_for_object_handle(object_handle)
        for state in states:
            spec = state.state_spec
            if spec.type == bool:
                name = spec.name
                value = cast(bool, state.value)
                target_value = not value
                action = self.get_boolean_action(name, target_value)
                action_available = action is not None
                action_enabled = False
                error = None
                if action_available:
                    result = self.can_execute_action(
                        name, target_value, object_handle
                    )
                    action_enabled = result.success
                    error = result.error_message
                actions.append(
                    BooleanAction(
                        state_spec=spec,
                        current_value=state.value,
                        target_value=target_value,
                        available=action_available,
                        enabled=action_enabled,
                        error=error,
                    )
                )
        return actions

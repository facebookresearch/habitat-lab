#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum
from typing import Any, Optional
from habitat.datasets.utils import HabitatSimActions
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.key_mapping import KeyCode
import magnum as mn
import numpy as np

from habitat_hitl.environment.controllers.controller_abc import GuiController
from habitat_baselines.utils.common import get_num_actions
from habitat.core.spaces import ActionSpace

from habitat_llm.tools.motor_skills.skill import SkillPolicy
from habitat_llm.tools.motor_skills.pick.oracle_point_pick_skill import OraclePointPickSkill
from habitat_llm.tools.motor_skills.place.oracle_place_skill import OraclePlaceSkill
from habitat_llm.agent.env.environment_interface import EnvironmentInterface

# Method to find action range
# An equivalent method exists in habitat-lab but its buggy
def find_action_range(action_space: ActionSpace, search_key: str) -> tuple[int, int]:
    """
    Returns the start and end indices of an action key in the action tensor. If
    the key is not found, a Value error will be thrown.
    :param action_space: The set of all actions we consider.
    :param search_key: The action for which we want to find the range.
    """

    start_idx = 0
    found = False
    end_idx = get_num_actions(action_space[search_key])
    for k in action_space:
        if k == search_key:
            found = True
            break
        start_idx += get_num_actions(action_space[k])
    if not found:
        raise ValueError(f"Could not find {search_key} action in {action_space}")
    return start_idx, start_idx + end_idx

class SkillType(Enum):
    PICK=0
    PLACE=1

class SpotSkillsController(GuiController):
    def __init__(
        self,
        agent_index: int,
        articulated_agent: Any,     # TODO: Type
        gui_input: GuiInput,        # TODO: Not the controller's responsibility.
        # Skills
        skill_config: Any,          # See `hitl_skills.yaml`. TODO: Unknown fields.
        observation_space: Any,     # TODO: Unknown origin, type and configuration.
        action_space: Any,          # TODO: Unknown origin, type and configuration.
        batch_size: int,            # TODO: Unknown signification.
        env: EnvironmentInterface,  # TODO: Incompatible with non-partnr code.
        # Actions
        num_actions: int,           # TODO: Unknown.
        base_vel_action_idx: int,   # TODO: Unknown.
        num_base_vel_actions: int,  # TODO: Unknown.
        turn_scale: float,
    ):
        super().__init__(
            agent_idx=agent_index,
            is_multi_agent=False,   # TODO: Not the controller's responsibility.
            gui_input=gui_input,    # TODO: Not the controller's responsibility.
        )
        self._num_actions = num_actions
        self._articulated_agent = articulated_agent
        self._base_vel_action_idx = base_vel_action_idx
        self._num_base_vel_actions = num_base_vel_actions
        self._turn_scale = turn_scale

        # Initialize Actions
        self._actions = np.zeros((num_actions,))

        # Initialize Skills
        self._pick_skill = OraclePointPickSkill(
            skill_config.pick_skill_config,
            observation_space=observation_space,
            action_space=action_space,
            batch_size=batch_size,
            env=env,
            agent_uid = agent_index,
        )
        self._place_skill = OraclePlaceSkill(
            skill_config.place_skill_config,
            observation_space=observation_space,
            action_space=action_space,
            batch_size=batch_size,
            env=env,
            agent_uid = agent_index,
        )
        self._skills: dict[SkillType, SkillPolicy] = {
            SkillType.PICK:self._pick_skill,
            SkillType.PLACE:self._place_skill,
        }
        self._active_skill: Optional[SkillType] = None

    def _update(self):
        # TODO: Separate update and act calls. Include deltatime.
        if self._gui_input.get_key_down(KeyCode.P):
            self._active_skill = SkillType.PICK

    def act(self, obs, env):
        self._update()
        self._actions = np.zeros((self._num_actions,))  # TODO: Unknown behaviour.

        if self._active_skill is None:
            return self._actions
        
        skill = self._skills[self._active_skill]
        skill_action, hxs = skill.act(
            observations=obs,               # TODO: Unknown type and requirements.
            rnn_hidden_states=None,         # TODO: Unknown type and behaviour. Is this nullable?
            prev_actions=self._actions,     # TODO: Unknown type, behaviour and requirements.
            masks=None,                     # TODO: Unknown type and significance.
            cur_batch_idx=[0],              # TODO: Unknown behaviour.
            deterministic=True,             # TODO: Unknown behaviour.
        )
        self._actions = skill_action
        
        return self._actions

    def on_environment_reset(self):
        self._actions = np.zeros((self._num_actions,))
        batch_idxs = [0]    # TODO: Unknown value/behaviour.
        for skill in self._skills.values():
            skill.reset(batch_idxs)

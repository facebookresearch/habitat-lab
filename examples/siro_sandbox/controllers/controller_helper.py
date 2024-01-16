#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, List, Optional

import numpy as np

from .baselines_controller import (
    MultiAgentBaselinesController,
    SingleAgentBaselinesController,
)
from .controller_abc import Controller
from .gui_controller import GuiHumanoidController, GuiRobotController

if TYPE_CHECKING:
    from omegaconf import DictConfig

    import habitat
    from habitat.core.environments import GymHabitatEnv


class ControllerHelper:
    """ControllerHelper is a wrapper around the habitat env that allows for multiple agent controllers to be used."""

    def __init__(
        self,
        gym_habitat_env: "GymHabitatEnv",
        config: "DictConfig",
        hitl_config,
        gui_input,
        recorder,
    ):
        self._gym_habitat_env: GymHabitatEnv = gym_habitat_env
        self._env: habitat.Env = gym_habitat_env.unwrapped.habitat_env
        self._gui_controlled_agent_index = (
            config.habitat_hitl.gui_controlled_agent.agent_index
        )

        self.n_agents: int = len(self._env._sim.agents_mgr)  # type: ignore[attr-defined]
        self.n_user_controlled_agents: int = (
            0 if self._gui_controlled_agent_index is None else 1
        )
        self.n_policy_controlled_agents: int = (
            self.n_agents - self.n_user_controlled_agents
        )
        is_multi_agent: bool = self.n_agents > 1

        if self.n_agents > 2:
            raise ValueError("ControllerHelper only supports 1 or 2 agents.")

        self.controllers: List[Controller] = []
        if self.n_agents == self.n_policy_controlled_agents:
            # all agents are policy controlled
            if not is_multi_agent:
                # single agent case
                self.controllers.append(
                    SingleAgentBaselinesController(
                        0,
                        is_multi_agent,
                        config,
                        self._gym_habitat_env,
                    )
                )
            else:
                # multi agent case (2 agents)
                self.controllers.append(
                    MultiAgentBaselinesController(
                        is_multi_agent,
                        config,
                        self._gym_habitat_env,
                    )
                )
        else:
            # one agent is gui controlled and the rest (if any) are policy controlled
            agent_name: str = self._env.sim.habitat_config.agents_order[
                self._gui_controlled_agent_index
            ]
            articulated_agent_type: str = self._env.sim.habitat_config.agents[
                agent_name
            ].articulated_agent_type

            gui_agent_controller: Controller
            if articulated_agent_type == "KinematicHumanoid":
                gui_agent_controller = GuiHumanoidController(
                    agent_idx=self._gui_controlled_agent_index,
                    is_multi_agent=is_multi_agent,
                    gui_input=gui_input,
                    env=self._env,
                    walk_pose_path=hitl_config.walk_pose_path,
                    lin_speed=hitl_config.gui_controlled_agent.lin_speed,
                    ang_speed=hitl_config.gui_controlled_agent.ang_speed,
                    recorder=recorder.get_nested_recorder("gui_humanoid"),
                )
            else:
                gui_agent_controller = GuiRobotController(
                    agent_idx=self._gui_controlled_agent_index,
                    is_multi_agent=is_multi_agent,
                    gui_input=gui_input,
                )
            self.controllers.append(gui_agent_controller)

            if is_multi_agent:
                self.controllers.append(
                    SingleAgentBaselinesController(
                        0 if self._gui_controlled_agent_index == 1 else 1,
                        is_multi_agent,
                        config,
                        self._gym_habitat_env,
                    )
                )

    def get_gui_agent_controller(self) -> Optional[Controller]:
        if self._gui_controlled_agent_index is None:
            return None

        return self.controllers[0]

    def get_gui_controlled_agent_index(self) -> Optional[int]:
        return self._gui_controlled_agent_index

    def update(self, obs):
        actions = []

        for controller in self.controllers:
            controller_action = controller.act(obs, self._env)
            actions.append(controller_action)

        if len(self.controllers) == 1:
            action = actions.pop()
        elif len(self.controllers) == 2:
            # controllers don't necessarily act in the same order as the their agent index
            # so we need to sort the actions by agent index
            controlled_agent_idxs = [controller._agent_idx for controller in self.controllers]  # type: ignore[attr-defined]
            actions = [
                action
                for _, action in sorted(zip(controlled_agent_idxs, actions))
            ]
            action = np.concatenate(actions, dtype=np.float32)
        else:
            raise ValueError(
                "ControllerHelper only supports up to 2 controllers."
            )

        return action

    def on_environment_reset(self):
        for controller in self.controllers:
            controller.on_environment_reset()

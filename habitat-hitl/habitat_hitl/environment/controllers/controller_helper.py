#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, List

import numpy as np

import habitat.gym.gym_wrapper as gym_wrapper
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_hitl.environment.controllers.baselines_controller import (
    MultiAgentBaselinesController,
    SingleAgentBaselinesController,
    clean_dict,
)
from habitat_hitl.environment.controllers.controller_abc import Controller
from habitat_hitl.environment.controllers.gui_controller import (
    GuiHumanoidController,
    GuiRobotController,
)

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
        self._hitl_config = hitl_config
        self._gym_habitat_env: GymHabitatEnv = gym_habitat_env
        self._env: habitat.Env = gym_habitat_env.unwrapped.habitat_env
        self.n_agents: int = len(self._env._sim.agents_mgr)  # type: ignore[attr-defined]
        self.n_user_controlled_agents: int = len(
            hitl_config.gui_controlled_agents
        )
        assert self.n_user_controlled_agents <= self.n_agents
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
            # some agents are gui controlled and the rest (if any) are policy controlled
            for agent_index in range(
                len(self._env.sim.habitat_config.agents_order)
            ):
                gui_controlled_agent_config = (
                    self._find_gui_controlled_agent_config(agent_index)
                )
                if gui_controlled_agent_config:
                    agent_name: str = (
                        self._env.sim.habitat_config.agents_order[agent_index]
                    )
                    articulated_agent_type: str = (
                        self._env.sim.habitat_config.agents[
                            agent_name
                        ].articulated_agent_type
                    )

                    gui_agent_controller: Controller
                    if articulated_agent_type == "KinematicHumanoid":
                        gui_agent_controller = GuiHumanoidController(
                            agent_idx=agent_index,
                            is_multi_agent=is_multi_agent,
                            gui_input=gui_input,
                            env=self._env,
                            walk_pose_path=hitl_config.walk_pose_path,
                            lin_speed=gui_controlled_agent_config.lin_speed,
                            ang_speed=gui_controlled_agent_config.ang_speed,
                            recorder=recorder.get_nested_recorder(
                                "gui_humanoid"
                            ),
                        )
                    elif articulated_agent_type == "SpotRobot":
                        agent_k = f"agent_{agent_index}"
                        original_action_space = clean_dict(
                            self._gym_habitat_env.original_action_space,
                            agent_k,
                        )
                        action_space = gym_wrapper.create_action_space(
                            original_action_space
                        )

                        (
                            base_vel_action_idx,
                            base_vel_action_end_idx,
                        ) = find_action_range(
                            original_action_space, "_base_velocity"
                        )

                        assert len(action_space.shape) == 1
                        num_actions = action_space.shape[0]

                        articulated_agent = self._env._sim.agents_mgr[agent_index].articulated_agent  # type: ignore[attr-defined]

                        # sloppy: derive turn scale. This is the change in yaw (in radians) corresponding to a base ang vel action of 1.0. See also Habitat-lab BaseVelAction.
                        turn_scale = (
                            config.habitat.simulator.ctrl_freq
                            / config.habitat.task.actions[
                                f"{agent_k}_base_velocity"
                            ].ang_speed
                        )

                        gui_agent_controller = GuiRobotController(
                            agent_idx=agent_index,
                            is_multi_agent=is_multi_agent,
                            gui_input=gui_input,
                            articulated_agent=articulated_agent,
                            num_actions=num_actions,
                            base_vel_action_idx=base_vel_action_idx,
                            num_base_vel_actions=base_vel_action_end_idx
                            - base_vel_action_idx,
                            turn_scale=turn_scale,
                        )
                    else:
                        raise ValueError(
                            f"articulated agent type {articulated_agent_type} not supported"
                        )

                    self.controllers.append(gui_agent_controller)

                else:
                    self.controllers.append(
                        SingleAgentBaselinesController(
                            agent_index,
                            is_multi_agent,
                            config,
                            self._gym_habitat_env,
                        )
                    )

    def _find_gui_controlled_agent_config(self, agent_index):
        for (
            gui_controlled_agent_config
        ) in self._hitl_config.gui_controlled_agents:
            if gui_controlled_agent_config.agent_index == agent_index:
                return gui_controlled_agent_config
        return None

    def get_gui_agent_controllers(self) -> List[Controller]:
        """
        Return list of controllers indexed by user index. Beware the difference between user index and agent index. For example, user 0 may control agent 1.
        """
        gui_agent_controllers = []
        for (
            gui_controlled_agent_config
        ) in self._hitl_config.gui_controlled_agents:
            gui_agent_controllers.append(
                self.controllers[gui_controlled_agent_config.agent_index]
            )
        return gui_agent_controllers

    def get_all_agent_controllers(self) -> List[Controller]:
        """
        Return a list of controllers indexed by agent index.
        """
        return self.controllers

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

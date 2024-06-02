#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This controller assumes you are using a habitat-llm Agent downstream
# code for interface followed by a habitat-llm Agent will be released in the future

import logging
import threading
from typing import Any, Dict, List, Union

import numpy as np
from habitat_llm.agent import Agent
from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.planner.llm_planner import LLMPlanner
from habitat_llm.utils import fix_config, setup_config
from hydra.utils import instantiate
from omegaconf import DictConfig

import habitat
import habitat.config
from habitat.core.environments import GymHabitatEnv
from habitat_hitl.environment.controllers.baselines_controller import (
    SingleAgentBaselinesController,
)


class LLMController(SingleAgentBaselinesController):
    """Controller for single LLM controlled agent."""

    def __init__(
        self,
        agent_idx: int,
        is_multi_agent: bool,
        config: DictConfig,
        gym_habitat_env: GymHabitatEnv,
        log_to_file: bool = False,
    ):
        self._config = config
        self._is_multi_agent = is_multi_agent
        self._gym_habitat_env = gym_habitat_env
        self._habitat_env = gym_habitat_env.unwrapped.habitat_env
        self._agent_idx = agent_idx
        # TODO: gather this from config
        self._agent_action_length = 36
        self._thread: Union[None, threading.Thread] = None
        self._low_level_actions: Union[None, dict, np.ndarray] = {}
        self._task_done = False
        self._iter = 0
        self._skip_iters = 0
        if log_to_file:
            import datetime

            now = datetime.datetime.now()
            logging.basicConfig(
                filename=f"./act_timing_{now:%Y-%m-%d}_{now:%H-%M}.log",
                filemode="a",
                format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
                level=logging.DEBUG,
                force=True,
            )

        with habitat.config.read_write(self._config):
            fix_config(self._config)
        seed = 47668090
        with habitat.config.read_write(self._config):
            self._config = setup_config(self._config, seed)
        self.planner: LLMPlanner = None
        self.environment_interface: EnvironmentInterface = None

        # NOTE: this is creating just one agent. Habitat-LLM has code for creating
        # multiple processes/agents in one go. I am only prototyping single process, as
        # I assume the onus of creating multiple processes is on the user/hitl_driver,
        # and this code will be called once per Sim instantiation
        self.initialize_environment_interface()
        self.initialize_planner()
        self.info: Dict[str, Any] = {}
        self._human_action_history: List[Any] = []

    def initialize_planner(self):
        # NOTE: using instantiate here, but given this is planning for a single agent
        # always will this ever be an option of Centralized vs Decentralized? Maybe
        # DAG...?
        # NOTE: assuming use of DecentralizedLLMPlanner here
        planner = instantiate(self._config.evaluation.agents.agent_0.planner)
        self.planner = planner(env_interface=self.environment_interface)
        self.planner.agents = self.initialize_agents(
            self._config.evaluation.agents
        )

    def initialize_agents(self, agent_configs):
        agents = []
        for _, agent_conf in agent_configs.items():
            # Instantiate the agent
            agent = Agent(
                agent_conf.uid, agent_conf.config, self.environment_interface
            )
            agents.append(agent)
        return agents

    def initialize_environment_interface(self):
        self.environment_interface = EnvironmentInterface(
            self._config, gym_habitat_env=self._gym_habitat_env
        )

    def on_environment_reset(self):
        # NOTE: the following ONLY resets self._test_recurrent_hidden_states,
        # self._prev_actions and self._not_done_masks
        # super().on_environment_reset()
        self.environment_interface.reset_environment(reset_habitat=False)
        self.planner.reset()
        if self._thread is not None:
            self._thread.join()
            self._thread = None  # noqa: F841
            self._low_level_actions = {}

        self.current_instruction = (
            self.environment_interface.hab_env.current_episode.instruction
        )
        print(f"Instruction: {self.current_instruction}")
        self._iter = 0

    def _on_pick(self, _e: Any = None):
        action = {
            "action": "PICK",
            "object_id": _e.object_id,
            "object_handle": _e.object_handle,
        }

        self._human_action_history.append(action)

    def _on_place(self, _e: Any = None):
        action = {
            "action": "PLACE",
            "object_id": _e.object_id,
            "object_handle": _e.object_handle,
            "receptacle_id": _e.receptacle_id,
            # "receptacle_name": self.environment_interface.world_graph.get_node_from_sim_handle(
            #     get_obj_from_id(self.environment_interface.sim, _e.receptacle_id).handle
            # ),
        }

        self._human_action_history.append(action)

    def _on_open(self, _e: Any = None):
        action = {
            "action": "OPEN",
            "object_id": _e.object_id,
            "object_handle": _e.object_handle,
        }

        self._human_action_history.append(action)

    def _on_close(self, _e: Any = None):
        action = {
            "action": "CLOSE",
            "object_id": _e.object_id,
            "object_handle": _e.object_handle,
        }

        self._human_action_history.append(action)

    def _act(self, observations, *args, **kwargs):
        # NOTE: this is where the LLM magic happens, the agent is given the observations
        # and it returns the actions for the agent
        (
            self._low_level_actions,
            planner_info,
            self._task_done,
        ) = self.planner.get_next_action(
            self.current_instruction,
            observations,
            self.environment_interface.world_graph,
            verbose=True,
        )
        return

    def act(self, observations, *args, **kwargs):
        # NOTE: update the world state to reflect the new observations
        # TODO: might need a lock on world-state here?
        self.environment_interface.update_world_state(
            observations, disable_logging=True
        )
        # update agent state history
        while self._human_action_history:
            action = self._human_action_history.pop(0)
            if action["action"] == "PICK":
                object_name = self.environment_interface.world_graph.get_node_from_sim_handle(
                    action["object_handle"]
                ).name
                self.environment_interface.agent_state_history[1].append(
                    f"Agent picked up {object_name}"
                )
            elif action["action"] == "PLACE":
                object_name = self.environment_interface.world_graph.get_node_from_sim_handle(
                    action["object_handle"]
                ).name
                self.environment_interface.agent_state_history[1].append(
                    f"Agent placed {object_name} in {action['receptacle_id']}"
                )
        if self._iter < self._skip_iters or self._task_done:
            self._iter += 1
            return np.zeros(self._agent_action_length)
        low_level_actions = np.zeros(self._agent_action_length)

        if self._thread is None or not self._thread.is_alive():
            if self._low_level_actions != {}:
                low_level_actions = self._low_level_actions[
                    str(self._agent_idx)
                ][:-248]
            self._thread = self._thread = threading.Thread(
                target=self._act, args=(observations,), kwargs=kwargs
            )
            self._thread.start()

        return low_level_actions

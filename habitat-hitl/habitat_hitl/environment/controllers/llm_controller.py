#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This controller assumes you are using a habitat-llm Agent downstream
# code for interface followed by a habitat-llm Agent will be released in the future

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Union

import cv2
import numpy as np
from habitat_llm.agent import Agent
from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.planner.llm_planner import LLMPlanner
from habitat_llm.utils import fix_config, setup_config
from habitat_llm.world_model import Furniture
from hydra.utils import instantiate
from omegaconf import DictConfig

import habitat
import habitat.config
from habitat.core.environments import GymHabitatEnv
from habitat.sims.habitat_simulator.sim_utilities import get_obj_from_id
from habitat_hitl.core.event import Event
from habitat_hitl.environment.controllers.baselines_controller import (
    SingleAgentBaselinesController,
)


class PlannerStatus(Enum):
    SUCCESS = 0
    FAILED = 1
    ERROR = 2


@dataclass
class AgentTerminationEvent:
    status: PlannerStatus
    message: str


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
        self._log: list = []
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
        self._planner_info: dict = {}

        # interfacing with HitL
        self._on_termination = Event()
        self._termination_reported = False

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
        if self._thread is not None:
            self._thread.join()
        self._thread = None
        self.environment_interface.reset_environment(reset_habitat=False)
        self.planner.reset()

        self.current_instruction = (
            self.environment_interface.hab_env.current_episode.instruction
        )
        self._iter = 0
        self._log = []
        self._termination_reported = False
        self._low_level_actions = {}
        self._task_done = False
        self._iter = 0
        self._skip_iters = 0
        self._log = []

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

    def push_user_actions_to_llm(self):
        # update agent state history
        while self._human_action_history:
            action = self._human_action_history.pop(0)
            object_name = None
            try:
                object_name = self.environment_interface.world_graph.get_node_from_sim_handle(
                    action["object_handle"]
                ).name
            except Exception as e:
                self._log.append(e)
                continue
            if action["action"] == "PICK":
                self.environment_interface.agent_state_history[1].append(
                    f"Agent picked up {object_name}"
                )
            elif action["action"] == "PLACE":
                furniture_name = "unknown furniture"
                if action["receptacle_id"] is not None:
                    receptacle_node = self.environment_interface.world_graph.get_node_from_sim_handle(
                        get_obj_from_id(
                            self.environment_interface.sim,
                            action["receptacle_id"],
                        ).handle
                    )
                    if receptacle_node is not None:
                        if isinstance(receptacle_node, Furniture):
                            furniture_name = receptacle_node.name
                        else:
                            furnitures = self.environment_interface.world_graph.get_neighbors_of_type(
                                receptacle_node, Furniture
                            )
                            if len(furnitures) > 0:
                                furniture_name = furnitures[0].name
                            else:
                                print(
                                    "Could not find furniture for receptacle: ",
                                    receptacle_node.sim_handle,
                                    " ",
                                    receptacle_node.name,
                                )
                    else:
                        print("Receptacle not found")
                self.environment_interface.agent_state_history[1].append(
                    f"Agent placed {object_name} in/on {furniture_name}"
                )
            elif action["action"] == "OPEN":
                self.environment_interface.agent_state_history[1].append(
                    f"Agent opened {object_name}"
                )
            elif action["action"] == "CLOSE":
                self.environment_interface.agent_state_history[1].append(
                    f"Agent closed {object_name}"
                )

    def _act(self, observations, *args, **kwargs):
        # NOTE: this is where the LLM magic happens, the agent is given the observations
        # and it returns the actions for the agent
        (
            self._low_level_actions,
            self._planner_info,
            self._task_done,
        ) = self.planner.get_next_action(
            self.current_instruction,
            observations,
            self.environment_interface.world_graph,
            verbose=True,
        )
        return

    def act(self, observations, debug_obs: bool = False, *args, **kwargs):
        # set the task as done and report it back
        if self._task_done and not self._termination_reported:
            if (
                self._planner_info["replanning_count"]
                >= self._planner_info["replanning_threshold"]
            ):
                self._on_termination.invoke(
                    AgentTerminationEvent(status=PlannerStatus.FAILED, message="replanning threshold exceeded")
                )
            elif "ConnectionError" in self._planner_info["prompts"][0]:
                self._on_termination.invoke(
                    AgentTerminationEvent(status=PlannerStatus.ERROR, message="LLM connection error")
                )
            else:
                self._on_termination.invoke(
                    AgentTerminationEvent(status=PlannerStatus.SUCCESS, message="")
                )
            self._termination_reported = True

        low_level_actions = np.zeros(self._agent_action_length)

        if debug_obs and "agent_1_head_rgb" in observations:
            rgb = observations["agent_1_head_rgb"]
            panoptic = observations["agent_1_head_panoptic"]
            cv2.imwrite(
                f"./visuals/agent_1/rgb_{self._iter}.png",
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                f"./visuals/agent_1/panoptic_{self._iter}.png", panoptic
            )
        if debug_obs and "agent_0_articulated_agent_arm_rgb" in observations:
            rgb = observations["agent_0_articulated_agent_arm_rgb"]
            panoptic = observations["agent_0_articulated_agent_arm_panoptic"]
            cv2.imwrite(
                f"./visuals/agent_0/rgb_{self._iter}.png",
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                f"./visuals/agent_0/panoptic_{self._iter}.png", panoptic
            )

        # planning logic when task is not done
        if not self._task_done:
            # update world-graph and action history
            # TODO: might need a lock on world-state here?
            self.environment_interface.update_world_state(
                observations, disable_logging=True
            )
            self.push_user_actions_to_llm()

            # read thread result and create thread if previous thread is done
            if self._thread is None or not self._thread.is_alive():
                if self._low_level_actions != {}:
                    low_level_actions = self._low_level_actions[
                        str(self._agent_idx)
                    ][
                        :-248
                    ]  # TODO: bad; fix this by reading action-space from config
                self._thread = self._thread = threading.Thread(
                    target=self._act, args=(observations,), kwargs=kwargs
                )
                self._thread.start()

        self._iter += 1
        return low_level_actions

from typing import Any, Dict, Union

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
    ):
        self._config = config
        self._is_multi_agent = is_multi_agent
        self._gym_habitat_env = gym_habitat_env
        self._habitat_env = gym_habitat_env.unwrapped.habitat_env
        self._agent_idx = agent_idx
        # TODO: gather this from config
        self._agent_action_length = 28

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

            # Make sure that its unique by adding to the set
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
        self.planner.reset()
        self.environment_interface.reset_environment()

        self.current_instruction = (
            self.environment_interface.hab_env.current_episode.instruction
        )

    def act(self, observations, *args, **kwargs):
        # NOTE: update the world state to reflect the new observations
        self.environment_interface.update_world_state(observations)

        # NOTE: this is where the LLM magic happens, the agent is given the observations
        # and it returns the actions for the agent
        # TODO: looping needed here until a physical low-level-action is returned
        low_level_actions: Union[dict, np.ndarray] = {}
        (
            low_level_actions,
            planner_info,
            task_done,
        ) = self.planner.get_next_action(
            self.current_instruction,
            observations,
            self.environment_interface.world_graph,
            verbose=True,
        )
        if low_level_actions:
            low_level_actions = low_level_actions[str(self._agent_idx)]
            # NOTE: truncating the action here, as this includes both Spot and Human actions
            low_level_actions = low_level_actions[:-248]
        else:
            low_level_actions = np.zeros(self._agent_action_length)
        return low_level_actions

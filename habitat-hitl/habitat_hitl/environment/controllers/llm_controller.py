from typing import Any, Dict

from habitat_llm.agent import Agent
from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.planner.llm_planner import LLMPlanner
from habitat_llm.utils import fix_config, setup_config
from hydra.utils import instantiate
from omegaconf import DictConfig

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
        super().__init__(agent_idx, is_multi_agent, config, gym_habitat_env)
        fix_config(config)
        seed = 47668090
        self.config = setup_config(config, seed)
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
        # always will this ever be an option of Centralized vs Decentralized? Maybe DAG...?
        self.planner = instantiate(self.config.evaluation.planner)
        self.planner.agents = self.initialize_agents(
            self.config.evaluation.agents
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
            self.config, gym_habitat_env=self._gym_habitat_env
        )

        # NOTE: this is to replicate initial call of  get_next_action, in
        # run_instruction() method. I am not sure why we do this initially?
        (
            _low_level_actions,
            _planner_info,
            _task_done,
        ) = self.planner.get_next_action(
            self.current_instruction,
            {},
            self.environment_interface.world_graph,
        )

    def on_environment_reset(self):
        # NOTE: the following ONLY resets self._test_recurrent_hidden_states,
        # self._prev_actions and self._not_done_masks
        super().on_environment_reset()
        self.planner.reset()
        self.environment_interface.reset()

        self.current_instruction = (
            self.environment_interface.hab_env.current_episode.instruction
        )

    def act(self, observations):
        # NOTE: update the world state to reflect the new observations
        self.environment_interface.update_world_state(observations)

        # NOTE: this is where the LLM magic happens, the agent is given the observations
        # and it returns the actions for the agent
        (
            low_level_actions,
            planner_info,
            task_done,
        ) = self.planner.get_next_action(
            self.current_instruction,
            observations,
            self.environment_interface.world_graph,
        )
        return low_level_actions

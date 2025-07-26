# Standard library
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# Third-party
import numpy as np
import torch
from gym.spaces import Box, Discrete
from gym.spaces import Dict as SpaceDict
from matplotlib import path
from omegaconf import DictConfig, OmegaConf

# Habitat & Habitat Baselines
import habitat
from habitat import Env
from habitat.config import read_write
from habitat.core.agent import Agent
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.geometry_utils import angle_between_quaternions, quaternion_from_coeff
import habitat_sim
from habitat_sim.utils.common import quat_to_angle_axis
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import batch_obs

# Local
import utils
from rl_distance_train import distance_policy, dataset, distance_policy_gt


class ImageNavShortestPathFollower(ShortestPathFollower):
    def __init__(
        self,
        sim: "HabitatSim",
        goal_radius: float,
        goal_pos: Union[List[float], np.ndarray],
        goal_rotation: Union[List[float], np.ndarray],
        return_one_hot: bool = True,
        stop_on_error: bool = True,
        turn_angle: float = 15,
    ):
        super().__init__(sim, goal_radius, return_one_hot, stop_on_error)
        self.goal_pos = goal_pos
        self.goal_rotation = quaternion_from_coeff(goal_rotation)
        self.done = False
        self.turn_angle = np.deg2rad(turn_angle)

    def get_next_action(self, goal_pos=None) -> Optional[Union[int, np.ndarray]]:
        best_action = super().get_next_action(self.goal_pos)

        if self.done or best_action == HabitatSimActions.stop:
            self.done = True

            current_q = self._sim.get_agent_state().rotation
            goal_q = self.goal_rotation

            rel_q = goal_q * current_q.inverse()
            angle, axis = quat_to_angle_axis(rel_q)
            signed_yaw = angle if axis[1] >= 0 else -angle
            signed_yaw = (signed_yaw + np.pi) % (2 * np.pi) - np.pi

            if abs(signed_yaw) < self.turn_angle / 2:
                return HabitatSimActions.stop
            elif signed_yaw < 0:
                return HabitatSimActions.turn_right
            else:
                return HabitatSimActions.turn_left

        else:
            return best_action

class PPOAgent(Agent):
    def __init__(self, config: DictConfig, model_weights: Dict) -> None:
        super().__init__()
        self.config = config

        with read_write(self.config):
            self.config.habitat.dataset.split = "val"

        with Env(config=self.config.habitat) as tmp_env:
            self.obs_space = tmp_env.observation_space
            self.act_space = tmp_env.action_space

        self.device = (
            torch.device(f"cuda:{self.config.habitat.simulator.habitat_sim_v0.gpu_device_id}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.hidden_size = self.config.habitat_baselines.rl.ppo.hidden_size

        random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore

        self.actor_critic = baseline_registry.get_policy(
            config.habitat_baselines.rl.policy.main_agent.name
        )
        self.actor_critic = self.actor_critic.from_config(
            self.config, 
            observation_space=self.obs_space, 
            action_space=self.act_space
        )
        self.actor_critic.to(self.device)
        self.actor_critic.eval()

        self.model_path = model_weights
        if self.model_path:
            self.actor_critic.load_state_dict(model_weights, strict=True)
        else:
            habitat.logger.error("Model checkpoint wasn't loaded, evaluating a random model.")

        self.test_recurrent_hidden_states: Optional[torch.Tensor] = None
        self.not_done_masks: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.test_recurrent_hidden_states = torch.zeros(
            1, self.actor_critic.net.num_recurrent_layers, self.hidden_size, device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

    def act(self, observations: Observations) -> Dict[str, int]:
        batch = batch_obs([observations], device=self.device)
        with torch.no_grad():
            action_data = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=True,
            )
            self.test_recurrent_hidden_states = action_data.rnn_hidden_states
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(action_data.actions)

        return action_data.env_actions[0][0].item()
    
    def get_observation_space(self) -> SpaceDict:
        """Return the observation space of the agent."""
        return self.obs_space

    @staticmethod
    def from_config(config_path: str) -> "PPOAgent":
        """Load the agent configuration from a file or DictConfig."""
        if config_path.endswith((".pth", ".pt")):
            config = torch.load(config_path, map_location="cpu", weights_only=False)
            model_weights = config['state_dict']
            config = OmegaConf.create(config['config'])
        else:
            raise ValueError("Unsupported file format. Use .yaml or .pth/.pt files.")

        return PPOAgent(config, model_weights)

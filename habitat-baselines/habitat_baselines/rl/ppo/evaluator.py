import abc
from typing import TYPE_CHECKING, Any, Dict, List, Set, Tuple, Union

import torch
from numpy import ndarray
from torch import Tensor

from habitat import VectorEnv
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr

if TYPE_CHECKING:
    from omegaconf import DictConfig


class Evaluator(abc.ABC):
    """
    Generic evaluator interface for evaluation loops over provided checkpoints.
    Extend for environment or project specific evaluation.
    """

    @abc.abstractmethod
    def evaluate_agent(
        self,
        agent: AgentAccessMgr,
        envs: VectorEnv,
        config: "DictConfig",
        checkpoint_index: int,
        step_id: int,
        writer: TensorboardWriter,
        device: torch.device,
        obs_transforms: List[ObservationTransformer],
        env_spec: EnvironmentSpec,
        rank0_keys: Set[str],
    ) -> None:
        """
        :param agent: Loaded policy to evaluate.
        :param envs: Vectorized environments to evaluate in.
        :param checkpoint_index: ID of the checkpoint (for logging).
        :param step_id: Training step of checkpoint (for logging)
        :param writer: Logger for recording metrics of evaluation.
        :param device: PyTorch device to use for evaluation
        :param obs_transforms: Observation transformations for the policy.
        :param env_spec: Environment action/observation spaces.
        :param rank0_keys: Info dictionary keys that should only be recorded on the 0th worker.

        Returns nothing. Evaluated metrics should be recorded via the `writer`.
        """


def pause_envs(
    envs_to_pause: List[int],
    envs: VectorEnv,
    test_recurrent_hidden_states: Tensor,
    not_done_masks: Tensor,
    current_episode_reward: Tensor,
    prev_actions: Tensor,
    batch: Dict[str, Tensor],
    rgb_frames: Union[List[List[Any]], List[List[ndarray]]],
) -> Tuple[
    VectorEnv,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Dict[str, Tensor],
    List[List[Any]],
]:
    # pausing self.envs with no new episode
    if len(envs_to_pause) > 0:
        state_index = list(range(envs.num_envs))
        for idx in reversed(envs_to_pause):
            state_index.pop(idx)
            envs.pause_at(idx)

        # indexing along the batch dimensions
        test_recurrent_hidden_states = test_recurrent_hidden_states[
            state_index
        ]
        not_done_masks = not_done_masks[state_index]
        current_episode_reward = current_episode_reward[state_index]
        prev_actions = prev_actions[state_index]

        for k, v in batch.items():
            batch[k] = v[state_index]

        if rgb_frames is not None:
            rgb_frames = [rgb_frames[i] for i in state_index]
        # actor_critic.do_pause(state_index)

    return (
        envs,
        test_recurrent_hidden_states,
        not_done_masks,
        current_episode_reward,
        prev_actions,
        batch,
        rgb_frames,
    )

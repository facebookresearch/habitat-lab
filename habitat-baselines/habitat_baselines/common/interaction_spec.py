from dataclasses import dataclass, field

import gym.spaces as spaces


@dataclass(frozen=True, kw_only=True)
class InteractionSpec:
    """
    :property action_space: The potentially flattened version of the environment action space.
    :property orig_action_space: The non-flattened version of the environment action space.
    """

    obs_space: spaces.Space
    env_action_space: spaces.Space
    orig_env_action_space: spaces.Space

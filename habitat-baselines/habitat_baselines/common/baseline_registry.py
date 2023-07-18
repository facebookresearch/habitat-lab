#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""BaselineRegistry is extended from habitat.Registry to provide
registration for trainer and policies, while keeping Registry
in habitat core intact.

Import the baseline registry object using

.. code:: py

    from habitat_baselines.common.baseline_registry import baseline_registry

Various decorators for registry different kind of classes with unique keys

-   Register a trainer: ``@baseline_registry.register_trainer``
-   Register a policy: ``@baseline_registry.register_policy``
"""

from typing import Optional

from habitat.core.registry import Registry


class BaselineRegistry(Registry):
    @classmethod
    def register_trainer(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a RL training algorithm to registry with key 'name'.

        Args:
            name: Key with which the trainer will be registered.
                If None will use the name of the class.

        """
        from habitat_baselines.common.base_trainer import BaseTrainer

        return cls._register_impl(
            "trainer", to_register, name, assert_type=BaseTrainer
        )

    @classmethod
    def get_trainer(cls, name):
        return cls._get_impl("trainer", name)

    @classmethod
    def register_policy(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a RL policy with :p:`name`.

        :param name: Key with which the policy will be registered.
            If :py:`None` will use the name of the class

        .. code:: py

            from habitat_baselines.rl.ppo.policy import Policy
            from habitat_baselines.common.baseline_registry import (
                baseline_registry
            )

            @baseline_registry.register_policy
            class MyPolicy(Policy):
                pass


            # or

            @baseline_registry.register_policy(name="MyPolicyName")
            class MyPolicy(Policy):
                pass

        """
        from habitat_baselines.rl.ppo.policy import Policy

        return cls._register_impl(
            "policy", to_register, name, assert_type=Policy
        )

    @classmethod
    def get_policy(cls, name: str):
        r"""Get the RL policy with :p:`name`."""
        return cls._get_impl("policy", name)

    @classmethod
    def register_obs_transformer(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        r"""Register a Observation Transformer with :p:`name`.

        :param name: Key with which the policy will be registered.
            If :py:`None` will use the name of the class

        .. code:: py

            from habitat_baselines.common.obs_transformers import ObservationTransformer
            from habitat_baselines.common.baseline_registry import (
                baseline_registry
            )

            @baseline_registry.register_policy
            class MyObsTransformer(ObservationTransformer):
                pass


            # or

            @baseline_registry.register_policy(name="MyTransformer")
            class MyObsTransformer(ObservationTransformer):
                pass

        """
        from habitat_baselines.common.obs_transformers import (
            ObservationTransformer,
        )

        return cls._register_impl(
            "obs_transformer",
            to_register,
            name,
            assert_type=ObservationTransformer,
        )

    @classmethod
    def get_obs_transformer(cls, name: str):
        r"""Get the Observation Transformer with :p:`name`."""
        return cls._get_impl("obs_transformer", name)

    @classmethod
    def register_auxiliary_loss(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        return cls._register_impl("aux_loss", to_register, name)

    @classmethod
    def get_auxiliary_loss(cls, name: str):
        return cls._get_impl("aux_loss", name)

    @classmethod
    def register_storage(cls, to_register=None, *, name: Optional[str] = None):
        """
        Registers data storage for storing data in the policy rollout in the
        trainer and then for fetching data batches for the updater.
        """

        return cls._register_impl("storage", to_register, name)

    @classmethod
    def get_storage(cls, name: str):
        return cls._get_impl("storage", name)

    @classmethod
    def register_agent_access_mgr(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        """
        Registers an agent access manager for the trainer to interface with. Usage:
        ```
        @baseline_registry.register_agent_access_mgr
        class ExampleAgentAccessMgr:
            pass
        ```
        or override the name with `name`.
        ```
        @baseline_registry.register_agent_access_mgr(name="MyAgentAccessMgr")
        class ExampleAgentAccessMgr:
            pass
        ```
        """
        from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr

        return cls._register_impl(
            "agent", to_register, name, assert_type=AgentAccessMgr
        )

    @classmethod
    def get_agent_access_mgr(cls, name: str):
        return cls._get_impl("agent", name)

    @classmethod
    def register_updater(cls, to_register=None, *, name: Optional[str] = None):
        """
        Registers a policy updater.
        """

        return cls._register_impl("updater", to_register, name)

    @classmethod
    def get_updater(cls, name: str):
        return cls._get_impl("updater", name)


baseline_registry = BaselineRegistry()

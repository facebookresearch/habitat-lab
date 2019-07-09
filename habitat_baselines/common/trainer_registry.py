from typing import Optional

from habitat.core.registry import Registry


class TrainRegistry(Registry):
    @classmethod
    def register_trainer(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a RL training algorithm to registry with key 'name'

        Args:
            name: Key with which the trainer will be registered.
                If None will use the name of the class

        """
        from habitat_baselines.common.base_trainer import BaseTrainer

        return cls._register_impl(
            "trainer", to_register, name, assert_type=BaseTrainer
        )

    @classmethod
    def get_trainer(cls, name):
        return cls._get_impl("trainer", name)


trainer_registry = TrainRegistry()

from abc import ABC, abstractmethod

from habitat_baselines.common.trainer_registry import trainer_registry


class BaseTrainer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def save_checkpoint(self, file_name):
        pass

    @abstractmethod
    def load_checkpoint(self):
        pass


def get_trainer(trainer_name, trainer_cfg):
    trainer = trainer_registry.get_trainer(trainer_name)
    assert trainer is not None, f"{trainer_name} is not supported"
    return trainer(trainer_cfg)


class BaseRLTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def save_checkpoint(self, file_name):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError

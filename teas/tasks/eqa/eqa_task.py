import teas
from teas.datasets import make_dataset
from teas.core.dataset import EQAEpisode
from typing import Any, Iterable, Tuple


class EQATask(teas.EmbodiedTask):
    def __init__(self, config: Any) -> None:
        self._config = config
        self._dataset = make_dataset(config.dataset.name,
                                     config=config.dataset)
        self._env = teas.TeasEnv(config=config)
        self.seed(config.seed)

    @staticmethod
    def _apply_episode(env_config: Any, episode: EQAEpisode) -> None:
        env_config.scene = episode.scene_file
        env_config.start_position = episode.start_position
        env_config.start_rotation = episode.start_rotation

    def episodes(self) -> Iterable[Tuple[EQAEpisode,
                                         teas.core.teas_env.TeasEnv]]:
        for i in range(len(self._dataset)):
            eqa_episode = self._dataset[i]
            self._apply_episode(self._config, eqa_episode)
            self._env.reconfigure(self._config)
            yield eqa_episode, self._env

    def seed(self, seed: int) -> None:
        self._env.seed(seed)

    def close(self) -> None:
        self._env.close()

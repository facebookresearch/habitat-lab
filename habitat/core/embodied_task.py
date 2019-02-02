from typing import Any, Type, Optional

from habitat.config import Config
from habitat.core.dataset import Episode, Dataset
from habitat.core.simulator import SensorSuite, Simulator


class EmbodiedTask:
    r"""Base class to keep whole Task specific logic added on top of simulator.
    """
    _config: Any
    _sim: Optional[Simulator]
    _dataset: Optional[Dataset]
    _sensor_suite: SensorSuite

    def __init__(
        self,
        config: Config,
        sim: Simulator,
        dataset: Optional[Dataset] = None,
        sensor_suite: SensorSuite = SensorSuite([]),
    ) -> None:
        self._config = config
        self._sim = sim
        self._dataset = dataset
        self._sensor_suite = sensor_suite

    def overwrite_sim_config(
        self, sim_config: Config, episode: Type[Episode]
    ) -> Config:
        r"""Returns updated simulator config with episode data such as a start
        state.
        """
        raise NotImplementedError

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

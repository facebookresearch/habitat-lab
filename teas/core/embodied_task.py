from typing import Any, Dict, Type, Optional

from teas.core.dataset import Episode, Dataset
from teas.core.simulator import Observation, SensorSuite, Simulator


class EmbodiedTask:
    r"""Base class to keep whole Task specific logic added on top of simulator.
    """
    _config: Any
    _simulator: Optional[Simulator]
    _dataset: Optional[Dataset]
    _sensor_suite: SensorSuite

    def __init__(self):
        self._config = None
        self._simulator = None
        self._dataset = None
        self._sensor_suite = SensorSuite([])

    def overwrite_sim_config(self, sim_config: Any,
                             episode: Type[Episode]) -> Any:
        r"""Returns updated simulator config with episode data such as a start
        state.
        """
        raise NotImplementedError

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    def get_reward(self, observations: Dict[str, Observation]) -> Any:
        raise NotImplementedError

from typing import Any, Dict, Type, Optional

from teas.core.dataset import Episode, Dataset
from teas.core.simulator import Observation, SensorSuite


class EmbodiedTask:

    def __init__(self):
        self._config: Any = None
        self._simulator: Any = None
        self._dataset: Optional[Dataset] = None
        self._sensor_suite: SensorSuite = None

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

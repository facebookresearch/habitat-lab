from habitat.config.default_structured_configs import (
    HabitatConfigPlugin,
    register_hydra_plugin,
)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)


def register_hydra_plugins():
    register_hydra_plugin(HabitatConfigPlugin)
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    # todo: also register search paths for hitl framework's configs

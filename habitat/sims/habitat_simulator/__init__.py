from habitat.core.registry import registry
from habitat.core.simulator import Simulator

# from habitat.sims.habitat_simulator.actions import (
#     HabitatSimV1ActionSpaceConfiguration,
# )


def _try_register_habitat_sim():
    try:
        import habitat_sim  # noqa: F401

        has_habitat_sim = True
    except ImportError as e:
        has_habitat_sim = False
        habitat_sim_import_error = e

    if has_habitat_sim:
        from habitat.sims.habitat_simulator.actions import (  # noqa: F401
            HabitatSimV1ActionSpaceConfiguration,
        )
    else:

        @registry.register_simulator(name="Sim-v0")
        class HabitatSimImportError(Simulator):
            def __init__(self, *args, **kwargs):
                raise habitat_sim_import_error

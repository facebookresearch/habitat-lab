#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import Simulator


def _try_register_habitat_sim():
    try:
        import habitat_sim

        has_habitat_sim = True
    except ImportError as e:
        has_habitat_sim = False
        habitat_sim_import_error = e

    if has_habitat_sim:
        from habitat.sims.habitat_simulator import HabitatSim
    else:

        @registry.register_simulator(name="Sim-v0")
        class HabitatSimImportError(Simulator):
            def __init__(self, *args, **kwargs):
                raise habitat_sim_import_error


def make_sim(id_sim, **kwargs):
    logger.info("initializing sim {}".format(id_sim))
    _sim = registry.get_simulator(id_sim)
    assert _sim is not None, "Could not find simulator with name {}".format(
        id_sim
    )
    return _sim(**kwargs)


_try_register_habitat_sim()

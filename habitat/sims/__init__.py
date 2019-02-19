#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.sims.registration import sim_registry, register_sim, make_sim

register_sim(
    id_sim="Sim-v0", entry_point="habitat.sims.habitat_simulator:HabitatSim"
)

__all__ = ["sim_registry", "register_sim", "make_sim"]

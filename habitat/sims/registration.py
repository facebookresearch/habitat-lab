#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.logging import logger
from habitat.core.registry import Registry, Spec


class SimSpec(Spec):
    def __init__(self, id_sim, entry_point):
        super().__init__(id_sim, entry_point)


class SimRegistry(Registry):
    def register(self, id_sim, **kwargs):
        if id_sim in self.specs:
            raise ValueError(
                "Cannot re-register sim"
                " specification with id: {}".format(id_sim)
            )
        self.specs[id_sim] = SimSpec(id_sim, **kwargs)


sim_registry = SimRegistry()


def register_sim(id_sim, **kwargs):
    sim_registry.register(id_sim, **kwargs)


def make_sim(id_sim, **kwargs):
    logger.info("initializing sim {}".format(id_sim))
    return sim_registry.make(id_sim, **kwargs)


def get_spec_sim(id_sim):
    return sim_registry.get_spec(id_sim)

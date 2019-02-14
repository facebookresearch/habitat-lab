#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.registry import Registry, Spec


class DatasetSpec(Spec):
    def __init__(self, id_dataset, entry_point):
        super().__init__(id_dataset, entry_point)


class DatasetRegistry(Registry):
    def register(self, id_dataset, **kwargs):
        if id_dataset in self.specs:
            raise ValueError(
                "Cannot re-register dataset  specification with id: {}".format(
                    id_dataset
                )
            )
        self.specs[id_dataset] = DatasetSpec(id_dataset, **kwargs)


dataset_registry = DatasetRegistry()


def register_dataset(id_dataset, **kwargs):
    dataset_registry.register(id_dataset, **kwargs)


def make_dataset(id_dataset, **kwargs):
    return dataset_registry.make(id_dataset, **kwargs)


def get_spec_dataset(id_dataset):
    return dataset_registry.get_spec(id_dataset)

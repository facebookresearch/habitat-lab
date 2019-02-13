#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any


def load(name: str) -> type:
    import pkg_resources

    entry_point = pkg_resources.EntryPoint.parse("x={}".format(name))
    result = entry_point.resolve()
    return result


class Spec:
    def __init__(self, id: str, entry_point: str, **kwargs: Any) -> None:
        self.id = id
        self._entry_point = entry_point

    def make(self, **kwargs: Any) -> Any:
        return load(self._entry_point)(**kwargs)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.id)


class Registry:
    def __init__(self):
        self.specs = {}

    def make(self, id: str, **kwargs: Any) -> Any:
        spec = self.get_spec(id)
        return spec.make(**kwargs)

    def all(self) -> Any:
        return self.specs.values()

    def get_spec(self, id: str) -> Spec:
        spec = self.specs.get(id, None)
        if spec is None:
            raise KeyError(
                "No registered specification with id: {}".format(id)
            )
        return spec

    def register(self, id: str, **kwargs: Any) -> None:
        raise NotImplementedError

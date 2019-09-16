#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Registry is central source of truth in Habitat.

Taken from Pythia, it is inspired from Redux's concept of global store.
Registry maintains mappings of various information to unique keys. Special
functions in registry can be used as decorators to register different kind of
classes.

Import the global registry object using

.. code:: py

    from habitat.core.registry import registry

Various decorators for registry different kind of classes with unique keys

-   Register a task: :py:`@registry.register_task`
-   Register a simulator: :py:`@registry.register_simulator`
-   Register a sensor: :py:`@registry.register_sensor`
-   Register a measure: :py:`@registry.register_measure`
-   Register a dataset: :py:`@registry.register_dataset`
"""

import collections
from typing import Optional

from habitat.core.utils import Singleton


class Registry(metaclass=Singleton):
    mapping = collections.defaultdict(dict)

    @classmethod
    def _register_impl(cls, _type, to_register, name, assert_type=None):
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(
                    to_register, assert_type
                )

            cls.mapping[_type][
                to_register.__name__ if name is None else name
            ] = to_register

            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def register_task(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a task to registry with key :p:`name`

        :param name: Key with which the task will be registered.
            If :py:`None` will use the name of the class

        .. code:: py

            from habitat.core.registry import registry
            from habitat.core.embodied_task import EmbodiedTask

            @registry.register_task
            class MyTask(EmbodiedTask):
                pass


            # or

            @registry.register_task(name="MyTaskName")
            class MyTask(EmbodiedTask):
                pass

        """
        from habitat.core.embodied_task import EmbodiedTask

        return cls._register_impl(
            "task", to_register, name, assert_type=EmbodiedTask
        )

    @classmethod
    def register_simulator(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        r"""Register a simulator to registry with key :p:`name`

        :param name: Key with which the simulator will be registered.
            If :py:`None` will use the name of the class

        .. code:: py

            from habitat.core.registry import registry
            from habitat.core.simulator import Simulator

            @registry.register_simulator
            class MySimulator(Simulator):
                pass


            # or

            @registry.register_simulator(name="MySimName")
            class MySimulator(Simulator):
                pass

        """
        from habitat.core.simulator import Simulator

        return cls._register_impl(
            "sim", to_register, name, assert_type=Simulator
        )

    @classmethod
    def register_sensor(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a sensor to registry with key :p:`name`

        :param name: Key with which the sensor will be registered.
            If :py:`None` will use the name of the class
        """
        from habitat.core.simulator import Sensor

        return cls._register_impl(
            "sensor", to_register, name, assert_type=Sensor
        )

    @classmethod
    def register_measure(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a measure to registry with key :p:`name`

        :param name: Key with which the measure will be registered.
            If :py:`None` will use the name of the class
        """
        from habitat.core.embodied_task import Measure

        return cls._register_impl(
            "measure", to_register, name, assert_type=Measure
        )

    @classmethod
    def register_dataset(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a dataset to registry with key :p:`name`

        :param name: Key with which the dataset will be registered.
            If :py:`None` will use the name of the class
        """
        from habitat.core.dataset import Dataset

        return cls._register_impl(
            "dataset", to_register, name, assert_type=Dataset
        )

    @classmethod
    def register_action_space_configuration(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        r"""Register a action space configuration to registry with key :p:`name`

        :param name: Key with which the action space will be registered.
            If :py:`None` will use the name of the class
        """
        from habitat.core.simulator import ActionSpaceConfiguration

        return cls._register_impl(
            "action_space_config",
            to_register,
            name,
            assert_type=ActionSpaceConfiguration,
        )

    @classmethod
    def _get_impl(cls, _type, name):
        return cls.mapping[_type].get(name, None)

    @classmethod
    def get_task(cls, name):
        return cls._get_impl("task", name)

    @classmethod
    def get_simulator(cls, name):
        return cls._get_impl("sim", name)

    @classmethod
    def get_sensor(cls, name):
        return cls._get_impl("sensor", name)

    @classmethod
    def get_measure(cls, name):
        return cls._get_impl("measure", name)

    @classmethod
    def get_dataset(cls, name):
        return cls._get_impl("dataset", name)

    @classmethod
    def get_action_space_configuration(cls, name):
        return cls._get_impl("action_space_config", name)


registry = Registry()

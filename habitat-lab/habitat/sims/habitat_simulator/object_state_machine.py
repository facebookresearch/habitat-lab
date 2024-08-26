#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This module implements a singleton state-machine architecture for representing and managing non-geometric object states via metadata manipulation. For example, tracking and manipulating state such as "powered on" or "clean vs dirty". This interface is intended to provide a foundation which can be extended for downstream applications."""

from collections import defaultdict
from typing import Any, Dict, List, Union

import magnum as mn

import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
from habitat.sims.habitat_simulator.debug_visualizer import (
    draw_object_highlight,
)
from habitat_sim.logging import logger
from habitat_sim.physics import ManagedArticulatedObject, ManagedRigidObject

##################################################
# Supporting utilities for getting and setting metadata values in ManagedObject "user_defined" Configurations.
##################################################


def get_state_of_obj(
    obj: Union[ManagedArticulatedObject, ManagedRigidObject],
    state_name: str,
) -> Any:
    """
    Try to get the specified state from an object's "object_states" user_defined metadata.

    :param obj: The ManagedObject.
    :param state_name: The name/key of the object state property to query.
    :return: The state value (variable type) or None if not found.
    """

    if "object_states" in obj.user_attributes.get_subconfig_keys():
        obj_states_config = obj.user_attributes.get_subconfig("object_states")
        if obj_states_config.has_value(state_name):
            return obj_states_config.get(state_name)
    return None


def set_state_of_obj(
    obj: Union[ManagedArticulatedObject, ManagedRigidObject],
    state_name: str,
    state_val: Any,
) -> None:
    """
    Set the specified state in an object's "object_states" user_defined metadata.

    :param obj: The ManagedObject.
    :param state_name: The name/key of the object state property to set.
    :param state_val: The value of the object state property to set.
    """

    user_attr = obj.user_attributes
    obj_state_config = user_attr.get_subconfig("object_states")
    obj_state_config.set(state_name, state_val)
    user_attr.save_subconfig("object_states", obj_state_config)


##################################################
# Object state machine implementation
##################################################


class ObjectStateSpec:
    """
    Abstract base class for object states specifications. Defines the API for inherited and extended states.

    An ObjectStateSpec is a singleton instance defining the interface and dynamics of a particular metadata state.

    Many ManagedObject instances can share an ObjectStateSpec, but there should be only one for each active Simulator since the state may compute and pivot on global internal caches and variables.
    """

    def __init__(self):
        # Each ObjectStateSpec should have a unique name string
        self.name = "AbstractState"
        # What type of data describes this state
        self.type = None
        # Human-readable name for display
        self.display_name = "Abstract State"
        # S list of semantic classes labels with pre-define membership in the state set. All objects in these classes are assumed to have this state, whether or not a value is defined in metadata.
        self.accepted_semantic_classes = []

    def is_affordance_of_obj(
        self, obj: Union[ManagedArticulatedObject, ManagedRigidObject]
    ) -> bool:
        """
        Determine whether or not an object instance can have this ObjectStateSpec by checking semantic class against the configured set.

        :param obj: The ManagedObject instance.
        :return: Whether or not the object has this state affordance.
        """

        # TODO: This is a placeholder until semantic_class can be officially supported or replaced by something else
        if (
            get_state_of_obj(obj, "semantic_class")
            in self.accepted_semantic_classes
        ):
            return True

        return False

    def update_state_context(self, sim: habitat_sim.Simulator) -> None:
        """
        Update internal state context independent of individual objects' states.

        :param sim: The Simulator instance.
        """

    def update_state(
        self,
        sim: habitat_sim.Simulator,
        obj: Union[ManagedArticulatedObject, ManagedRigidObject],
        dt: float,
    ) -> None:
        """
        Add state machine logic to modify the state of an object given access to the Simulator and timestep.
        Meant to be called from within the simulation or step loop to continuously update the state.

        :param sim: The Simulator instance.
        :param obj: The ManagedObject instance.
        :param dt: The timestep over which to update.
        """

    def default_value(self) -> Any:
        """
        If an object does not have a value for this state defined, return a default value.
        """

    def draw_context(
        self,
        debug_line_render: habitat_sim.gfx.DebugLineRender,
        camera_transform: mn.Matrix4,
    ) -> None:
        """
        Draw any context cues which are independent of individual objects' state.
        Meant to be called once per draw per ObjectStateSpec singleton.

        :param debug_line_render: The DebugLineRender instance for the Simulator.
        :param camera_transform: The Matrix4 camera transform.
        """

    def draw_state(
        self,
        obj: Union[ManagedArticulatedObject, ManagedRigidObject],
        debug_line_render: habitat_sim.gfx.DebugLineRender,
        camera_transform: mn.Matrix4,
    ) -> None:
        """
        Logic to draw debug lines visualizing this state for the object.

        :param obj: The ManagedObject instance.
        :param debug_line_render: The DebugLineRender instance for the Simulator.
        :param camera_transform: The Matrix4 camera transform.
        """


class BooleanObjectState(ObjectStateSpec):
    """
    Abstract ObjectStateSpec base class for boolean type states.
    Defines some standard handling for boolean states.
    """

    def __init__(self):
        self.name = "BooleanState"
        self.type = bool
        self.display_name = "Boolean State"
        self.display_name_true = "True"
        self.display_name_false = "False"

    def default_value(self) -> Any:
        """
        If an object does not have a value for this state defined, return a default value.
        """

        return True

    def draw_state(
        self,
        obj: Union[ManagedArticulatedObject, ManagedRigidObject],
        debug_line_render: habitat_sim.gfx.DebugLineRender,
        camera_transform: mn.Matrix4,
    ) -> None:
        """
        Logic to draw debug lines visualizing this state for the object.
        Draws a circle highlight around the object color by state value: green if True, red if False.

        :param obj: The ManagedObject instance.
        :param debug_line_render: The DebugLineRender instance for the Simulator.
        :param camera_transform: The Matrix4 camera transform.
        """

        obj_state = get_state_of_obj(obj, self.name)
        obj_state = self.default_value() if (obj_state is None) else obj_state

        color = mn.Color4.red()
        if obj_state:
            color = mn.Color4.green()

        draw_object_highlight(obj, debug_line_render, camera_transform, color)

    def toggle(
        self, obj: Union[ManagedArticulatedObject, ManagedRigidObject]
    ) -> bool:
        """
        Toggles a boolean state, returning the newly set value.

        :param obj: The ManagedObject instance.
        :return: The new value of the state.
        """

        cur_state = get_state_of_obj(obj, self.name)
        new_state = not cur_state
        set_state_of_obj(obj, self.name, new_state)
        return new_state


class ObjectIsClean(BooleanObjectState):
    """
    ObjectIsClean state specifies whether an object is clean or dirty.
    """

    def __init__(self):
        super().__init__()
        self.name = "is_clean"
        self.display_name = "Clean"
        self.display_name_true = "Clean"
        self.display_name_false = "Dirty"
        # TODO: set the semantic class membership list
        self.accepted_semantic_classes = []


class ObjectIsPoweredOn(BooleanObjectState):
    """
    State specifies whether an appliance object is powered on or off.
    """

    def __init__(self):
        super().__init__()
        self.name = "is_powered_on"
        self.display_name = "Powered On"
        self.display_name_true = "On"
        self.display_name_false = "Off"
        # TODO: set the semantic class membership list
        self.accepted_semantic_classes = []

    def default_value(self) -> Any:
        """
        Default value for power is off.
        """

        return False


class ObjectStateMachine:
    """
    Defines the logic for managing multiple states across all objects in the scene.
    """

    def __init__(self, active_states: List[ObjectStateSpec] = None) -> None:
        # a list of ObjectStateSpec singleton instances which are active in the current scene
        self.active_states = active_states if active_states is not None else []
        # map tracked objects to their set of state properties
        self.objects_with_states: Dict[
            str, List[ObjectStateSpec]
        ] = defaultdict(lambda: [])

    def initialize_object_state_map(self, sim: habitat_sim.Simulator) -> None:
        """
        Reset the objects_with_states dict and re-initializes it by parsing all objects from the scene and checking is_affordance_of_obj for all active ObjectStateSpecs.

        :param sim: The Simulator instance.
        """

        self.objects_with_states = defaultdict(lambda: [])
        all_objects = sutils.get_all_objects(sim)
        for obj in all_objects:
            self.register_object(obj)

    def register_object(
        self, obj: Union[ManagedArticulatedObject, ManagedRigidObject]
    ) -> None:
        """
        Register a single object in the 'objects_with_states' dict by checking 'is_affordance_of_obj' for all active ObjectStateSpecs.
        Use this when a new object is added to the scene and needs to be registered.

        :param obj: The ManagedObject instance to register.
        """

        for state in self.active_states:
            if state.is_affordance_of_obj(obj):
                self.objects_with_states[obj.handle].append(state)
                logger.debug(
                    f"registered state {state} for object {obj.handle}"
                )

    def update_states(self, sim: habitat_sim.Simulator, dt: float) -> None:
        """
        Update all tracked object states for a simulation step.

        :param sim: The Simulator instance.
        :param dt: The timestep over which to update continuous states. Typically the time between calls to this function.
        """

        # first update any state context
        for state in self.active_states:
            state.update_state_context(sim)
        # then update the individual object states
        for obj_handle, states in self.objects_with_states.items():
            if len(states) > 0:
                obj = sutils.get_obj_from_handle(sim, obj_handle)
                for state in states:
                    state.update_state(sim, obj, dt)

    def get_snapshot_dict(
        self, sim: habitat_sim.Simulator
    ) -> Dict[str, Dict[str, Any]]:
        """
        Scrape all active ObjectStateSpecs to collect a snapshot of the current state of all objects.

        :param sim: The Simulator instance for which to collect and return current object states.
        :return: The state snapshot as a Dict keyed by object state unique name, value is another dict mapping object instance handles to state values.

        Example:
            >>> {
            >>>     "is_powered_on": {
            >>>         "my_lamp.0001": True,
            >>>         "my_oven": False,
            >>>         ...
            >>>     },
            >>>     "is_clean": {
            >>>         "my_dish.0002:" False,
            >>>         ...
            >>>     },
            >>>     ...
            >>> }
        """
        snapshot: Dict[str, Dict[str, Any]] = defaultdict(lambda: {})
        for object_handle, states in self.objects_with_states.items():
            obj = sutils.get_obj_from_handle(sim, object_handle)
            for state in states:
                obj_state = get_state_of_obj(obj, state.name)
                snapshot[state.name][object_handle] = (
                    obj_state
                    if obj_state is not None
                    else state.default_value()
                )
        return dict(snapshot)

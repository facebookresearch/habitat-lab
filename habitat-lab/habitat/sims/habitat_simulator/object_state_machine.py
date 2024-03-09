#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Union

import magnum as mn

import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
from habitat.sims.habitat_simulator.debug_visualizer import (
    draw_object_highlight,
)
from habitat.sims.habitat_simulator.metadata_interface import MetadataInterface
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
    obj_state_config = None
    if "object_states" not in user_attr.get_subconfig_keys():
        obj_state_config = (
            habitat_sim._ext.habitat_sim_bindings.Configuration()
        )
    else:
        obj_state_config = user_attr.get_subconfig("object_states")
    obj_state_config.set(state_name, state_val)
    user_attr.save_subconfig("object_states", obj_state_config)


def set_all_object_semantic_classes_as_states(
    sim: habitat_sim.Simulator, metadata_interface: MetadataInterface
) -> None:
    """
    Sets the semantic class of each object as an object state 'semantic_class'.

    :param sim: The Simulator instance.
    :param metadata_interface: The MetadataInterface object used to map objects to their external semantics.
    """

    all_objs = sutils.get_all_objects(sim)
    for obj in all_objs:
        set_object_semantic_class_as_state(obj, metadata_interface)


def set_object_semantic_class_as_state(
    obj: Union[ManagedArticulatedObject, ManagedRigidObject],
    metadata_interface: MetadataInterface,
) -> None:
    """
    Sets the semantic class of an object as object state 'semantic_class'.

    :param obj: The ManagedObject instance.
    :param metadata_interface: The MetadataInterface object used to map objects to their external semantics.
    """

    obj_sem_class = metadata_interface.get_object_category(
        sutils.object_shortname_from_handle(obj.handle)
    )
    if obj_sem_class is not None:
        set_state_of_obj(obj, "semantic_class", obj_sem_class)


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
        # TODO: should be a broader set of categories
        self.accepted_semantic_classes = ["drinkware"]
        # store a map of object handles to local faucet points
        # TODO: should be another metadata pathway annotated for individual objects
        self.faucet_annotations: Dict[str, List[mn.Vector3]] = None
        # list of global positions which count as "faucets" for cleaning
        self.global_faucet_points = None
        # min distance from a faucet for an object to become "clean"
        self.faucet_range = 0.2

    def compute_global_faucet_points(
        self,
        sim: habitat_sim.Simulator,
        obj_faucet_points: Dict[str, List[mn.Vector3]],
    ) -> None:
        """
        Computes and caches a set of global 3D positions where an object could be cleaned from a map of objects to local faucet annotation points.

        :param sim: Simulator instance.
        :param obj_faucet_points: The map of object handles to local faucet point annotations.
        """

        self.global_faucet_points = []
        all_objects = sutils.get_all_objects(sim)
        for obj in all_objects:
            obj_hash = sutils.object_shortname_from_handle(obj.handle)
            if obj_hash in obj_faucet_points:
                self.global_faucet_points.append(
                    obj.transformation.transform_point(
                        obj_faucet_points[obj_hash]
                    )
                )
        print(f"compute_global_faucet_points: {self.global_faucet_points}")

    def update_state_context(self, sim: habitat_sim.Simulator) -> None:
        """
        Parse any faucet states from the active scene.

        :param sim: Simulator instance.
        """
        if self.global_faucet_points is None:
            self.compute_global_faucet_points(sim, self.faucet_annotations)

    def update_state(
        self,
        sim: habitat_sim.Simulator,
        obj: Union[ManagedArticulatedObject, ManagedRigidObject],
        dt: float,
    ) -> None:
        """
        Objects can become dirty by touching the floor and clean by getting close enough to a faucet.

        :param sim: The Simulator instance.
        :param obj: The ManagedObject instance.
        :param dt: The timestep over which to update.
        """

        cur_state = get_state_of_obj(obj, self.name)
        cur_state = self.default_value() if (cur_state is None) else cur_state
        if cur_state:
            # this is clean now, check if "on the floor"
            # NOTE: "on the floor" heuristic -> touching the "stage" includes walls, etc...
            cps = sim.get_physics_contact_points()
            for cp in cps:
                if (
                    cp.object_id_a == obj.object_id
                    or cp.object_id_b == obj.object_id
                ) and (
                    cp.object_id_a == habitat_sim.stage_id
                    or cp.object_id_b == habitat_sim.stage_id
                ):
                    set_state_of_obj(obj, self.name, False)
                    return
        else:
            # this is dirty, check if close enough to a faucet to clean
            # NOTE: uses L2 distance to faucet points to check if object can be cleaned
            # TODO: size_regularized_distance would be better
            obj_pos = obj.translation
            for point in self.global_faucet_points:
                dist = (obj_pos - point).length()
                if dist <= self.faucet_range:
                    set_state_of_obj(obj, self.name, True)
                    return

    def draw_context(
        self,
        debug_line_render: habitat_sim.gfx.DebugLineRender,
        camera_transform: mn.Matrix4,
    ) -> None:
        """
        Draw any context cues which are independent of individual objects' state.
        Draw the faucet points with yellow circle highlights.

        :param debug_line_render: The DebugLineRender instance for the Simulator.
        :param camera_transform: The Matrix4 camera transform.
        """

        if self.global_faucet_points is not None:
            for point in self.global_faucet_points:
                debug_line_render.draw_circle(
                    translation=point,
                    radius=self.faucet_range,
                    color=mn.Color4.yellow(),
                    normal=camera_transform.translation - point,
                )


class ObjectIsPoweredOn(BooleanObjectState):
    """
    State specifies whether an appliance object is powered on or off.
    """

    def __init__(self):
        super().__init__()
        self.name = "is_powered_on"
        # TODO: not an exaustive list
        self.accepted_semantic_classes = [
            "toaster",
            "tv",
            "range_hood",
            "coffee_maker",
            "floor_lamp",
            "washer_dryer",
        ]

    def default_value(self) -> Any:
        """
        Default value for power is off.
        """

        return False


class ObjectSateMachine:
    """
    Defines the logic for managing multiple states across all objects in the scene.
    """

    def __init__(self, active_states: List[ObjectStateSpec] = None) -> None:
        # a list of ObjectStateSpec singleton instances which are active in the current scene
        self.active_states = active_states if active_states is not None else []
        # map tracked objects to their set of state properies
        self.objects_with_states: Dict[str, List[ObjectStateSpec]] = {}

    def initialize_object_state_map(self, sim: habitat_sim.Simulator) -> None:
        """
        Reset the objects_with_states dict and re-initializes it by parsing all objects from the scene and checking is_affordance_of_obj for all active ObjectStateSpecs.

        :param sim: The Simulator instance.
        """

        self.objects_with_states = {}
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
                if obj.handle not in self.objects_with_states:
                    self.objects_with_states[obj.handle] = []
                self.objects_with_states[obj.handle].append(state)
                print(f"registered state {state} for object {obj.handle}")

    def update_states(self, sim: habitat_sim.Simulator, dt: float) -> None:
        """
        Update all tracked object states for a simulation step.

        :param sim: The Simulator instance.
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

    def report_tracked_states(self, sim: habitat_sim.Simulator) -> None:
        """
        Print a list of all tracked objects and their states.

        :param sim: The Simulator instance.
        """

        print("Object State Machine Report:")
        for obj_handle, states in self.objects_with_states.items():
            print(f"  - '{obj_handle}':")
            obj = sutils.get_obj_from_handle(sim, obj_handle)
            for state in states:
                print(
                    f"      {state.name} = {get_state_of_obj(obj, state.name)}"
                )

#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import magnum as mn

from habitat.sims.habitat_simulator.object_state_machine import (
    BooleanObjectState,
    ObjectStateMachine,
    get_state_of_obj,
    set_state_of_obj,
)
from habitat_sim import Simulator
from habitat_sim.physics import ManagedArticulatedObject, ManagedRigidObject
from habitat_sim.utils.settings import default_sim_settings, make_cfg


def test_state_getter_setter():
    """
    Test getting and setting state metadata to the object user_defined fields.
    """

    sim_settings = default_sim_settings.copy()
    hab_cfg = make_cfg(sim_settings)

    with Simulator(hab_cfg) as sim:
        obj_template_mngr = sim.get_object_template_manager()
        cube_obj_template = (
            obj_template_mngr.get_first_matching_template_by_handle("cube")
        )
        rom = sim.get_rigid_object_manager()
        new_obj = rom.add_object_by_template_handle(cube_obj_template.handle)

        test_state_values = ["string", 99, mn.Vector3(1.0, 2.0, 3.0)]

        assert get_state_of_obj(new_obj, "test_state") is None
        for test_state_val in test_state_values:
            set_state_of_obj(new_obj, "test_state", test_state_val)
            assert get_state_of_obj(new_obj, "test_state") == test_state_val


class TestObjectState(BooleanObjectState):
    def __init__(self):
        super().__init__()
        self.name = "TestState"
        # NOTE: This is contrived
        self.accepted_semantic_classes = ["test_class"]

    def update_state(
        self,
        sim: Simulator,
        obj: Union[ManagedArticulatedObject, ManagedRigidObject],
        dt: float,
    ) -> None:
        """
        Overwrite the update for a contrived unit test.
        Caches the time the object has been alive and when that time exceeds 1 second, sets the state to false.
        """
        time_alive = get_state_of_obj(obj, "time_alive")
        if time_alive is None:
            time_alive = 0
        time_alive += dt
        set_state_of_obj(obj, "time_alive", time_alive)
        if time_alive > 1.0:
            set_state_of_obj(obj, self.name, False)


def test_object_state_machine():
    """
    Test initializing and assigning a state to the state machine.
    Test contrived mechanics to proive and example of using the API.
    """

    # use an empty scene
    sim_settings = default_sim_settings.copy()
    hab_cfg = make_cfg(sim_settings)

    with Simulator(hab_cfg) as sim:
        obj_template_mngr = sim.get_object_template_manager()
        cube_obj_template = (
            obj_template_mngr.get_first_matching_template_by_handle("cube")
        )
        rom = sim.get_rigid_object_manager()
        new_obj = rom.add_object_by_template_handle(cube_obj_template.handle)

        # TODO: this is currently a contrived location to cache semantic state for category-based affordance logic.
        set_state_of_obj(new_obj, "semantic_class", "test_class")
        assert get_state_of_obj(new_obj, "semantic_class") == "test_class"

        # initialize the ObjectStateMachine
        osm = ObjectStateMachine(active_states=[TestObjectState()])
        osm.initialize_object_state_map(sim)

        # now the cube should be registered for TestObjectState because it has the correct semantic_class
        assert isinstance(osm.active_states[0], TestObjectState)
        assert new_obj.handle in osm.objects_with_states
        assert isinstance(
            osm.objects_with_states[new_obj.handle][0], TestObjectState
        )

        state_report_dict = osm.get_snapshot_dict(sim)
        assert "TestState" in state_report_dict
        assert new_obj.handle in state_report_dict["TestState"]
        assert (
            state_report_dict["TestState"][new_obj.handle]
            == TestObjectState().default_value()
        )

        # update the object state machine over time
        dt = 0.1
        while sim.get_world_time() < 2.0:
            sim.step_world(dt)
            osm.update_states(sim, dt)
            state_report_dict = osm.get_snapshot_dict(sim)
            if sim.get_world_time() < 1.0:
                assert state_report_dict["TestState"][new_obj.handle] == True
            else:
                assert state_report_dict["TestState"][new_obj.handle] == False

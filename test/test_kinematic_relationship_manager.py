#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp

import magnum as mn
import pytest

import habitat.datasets.rearrange.samplers.receptacle as hab_receptacle
import habitat.sims.habitat_simulator.sim_utilities as sutils
from habitat.sims.habitat_simulator.kinematic_relationship_manager import (
    KinematicRelationshipManager,
)
from habitat_sim import Simulator
from habitat_sim.utils.settings import default_sim_settings, make_cfg


@pytest.mark.skipif(
    not osp.exists("data/replica_cad/"),
    reason="Requires ReplicaCAD dataset.",
)
def test_kinematic_relationship_manager():
    """
    Test managing some kinematic states within ReplicaCAD "apt_0".
    """

    sim_settings = default_sim_settings.copy()
    sim_settings[
        "scene_dataset_config_file"
    ] = "data/replica_cad/replicaCAD.scene_dataset_config.json"
    sim_settings["scene"] = "apt_0"
    hab_cfg = make_cfg(sim_settings)

    with Simulator(hab_cfg) as sim:
        # construct the krm and initialize relationships
        krm = KinematicRelationshipManager(sim)

        # fetch some objects which known relationships in the scene
        table_object = sutils.get_obj_from_id(sim, 50)
        table_objects = [
            sutils.get_obj_from_id(sim, obj_id)
            for obj_id in [51, 52, 53, 54, 55, 102, 103]
        ]

        recs = hab_receptacle.find_receptacles(sim)

        # NOTE: below produces debugging output useful for examining the scene to understand what is happening
        # obj_id_to_handle = {
        #     obj.object_id: obj.handle for obj in sutils.get_all_objects(sim)
        # }
        # name_to_rec = {rec.unique_name: rec for rec in recs}
        # print("Objects:")
        # for obj_id, handle in obj_id_to_handle:
        #     print(f" - {obj_id} : {handle}")
        # print("\nReceptacles:")
        # for rec in recs:
        #     print(
        #         f" - {rec.unique_name} : {rec.parent_object_handle} | {rec.parent_link}"
        #     )

        # create some known mappings from objects to their supporting receptacles
        obj_to_rec_relations = {
            table_object.handle: "floor"  # explicitly test a floor parent relationship which should be skipped
        }
        # objects on the center table
        for table_obj in table_objects:
            obj_to_rec_relations[
                table_obj.handle
            ] = "frl_apartment_table_02_:0000|receptacle_aabb_Tbl2_Top1_frl_apartment_table_02"

        # initialize a KRM from the object to receptacle map
        krm.initialize_from_obj_to_rec_pairs(obj_to_rec_relations, recs)

        # check that the root parent was correctly registered
        root_parents = krm.relationship_graph.get_root_parents()
        assert len(root_parents) == 1
        assert root_parents[0] == table_object.object_id
        # Check that the relationships were registered correctly
        assert len(krm.relationship_graph.obj_to_children) == 1
        assert len(krm.relationship_graph.obj_to_parents) == len(table_objects)
        assert len(krm.relationship_graph.relation_types) == len(table_objects)
        for table_obj in table_objects:
            assert table_obj.object_id in krm.relationship_graph.obj_to_parents
            assert (
                table_obj.object_id
                in krm.relationship_graph.obj_to_children[
                    table_object.object_id
                ]
            )
        # check that the root parent snapshot is correct
        assert len(krm.prev_root_obj_state) == 1
        assert (
            krm.prev_root_obj_state[table_object.object_id]
            == table_object.transformation
        )
        # check that transforming the parent object works as expected
        offset = mn.Vector3(1.0, 2.0, 3.0)
        table_object.translate(offset)
        initial_table_obj_translations = {
            obj.object_id: obj.translation for obj in table_objects
        }
        krm.apply_relations()  # this should find the dirty transform and apply all cached relative transforms
        for table_obj in table_objects:
            assert (
                table_obj.translation
                == initial_table_obj_translations[table_obj.object_id] + offset
            )
        # test removing a parent relationship
        krm.relationship_graph.remove_obj_relations(table_objects[0].object_id)
        assert len(krm.relationship_graph.obj_to_children) == 1
        assert (
            len(krm.relationship_graph.obj_to_parents)
            == len(table_objects) - 1
        )
        assert (
            len(krm.relationship_graph.relation_types)
            == len(table_objects) - 1
        )
        for table_obj in table_objects[1:]:
            assert table_obj.object_id in krm.relationship_graph.obj_to_parents
            assert (
                table_obj.object_id
                in krm.relationship_graph.obj_to_children[
                    table_object.object_id
                ]
            )
        assert (
            table_objects[0].object_id
            not in krm.relationship_graph.obj_to_parents
        )
        assert (
            table_objects[0].object_id
            not in krm.relationship_graph.obj_to_children[
                table_object.object_id
            ]
        )
        # removed object is still in the snapshots until recomputed
        assert (
            table_objects[0].object_id
            in krm.prev_snapshot[table_object.object_id]
        )
        krm.update_snapshots()
        assert (
            table_objects[0].object_id
            not in krm.prev_snapshot[table_object.object_id]
        )

        # smoke test for utility function
        krm.relationship_graph.get_human_readable_relationship_forest(
            sim, do_print=False
        )

        # attempt an invalid parenting re-assignment (should trigger message and adjust the tree)
        test_child = list(krm.relationship_graph.obj_to_parents.keys())[0]
        test_parent = krm.relationship_graph.obj_to_parents[test_child]
        assert test_parent in krm.relationship_graph.obj_to_children
        assert (
            test_child in krm.relationship_graph.obj_to_children[test_parent]
        )
        # now add the previously removed parent as parent of this child
        new_test_parent = table_objects[0].object_id
        krm.relationship_graph.add_relation(new_test_parent, test_child, "on")
        assert (
            test_child
            not in krm.relationship_graph.obj_to_children[test_parent]
        )
        assert new_test_parent in krm.relationship_graph.obj_to_children
        assert (
            test_child
            in krm.relationship_graph.obj_to_children[new_test_parent]
        )
        assert (
            krm.relationship_graph.obj_to_parents[test_child]
            == new_test_parent
        )

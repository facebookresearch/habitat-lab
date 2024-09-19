#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This module implements a singleton manager class for tracking and applying kinematic relationships between objects in the simulation. It is meant to be instantiated upon Simulator init and then updated and applied as objects are moved, applying relative transformations down a parent->child kinematic tree."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import magnum as mn

import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
from habitat.core.logging import logger
from habitat.datasets.rearrange.samplers.receptacle import Receptacle


class RelationshipGraph:
    """
    Uses two dictionaries to simulate a bi-directional tree relationship between objects.

    NOTE: All 'obj' ints are assumed to be object ids so that links can be supported in the tree structure.

    NOTE: because links are parented explicitly to their parent AO, we don't allow them to be children in the relationship manager, only parents.
    """

    def __init__(self) -> None:
        # bi-directional relationship maps
        self.obj_to_children: Dict[int, List[int]] = {}
        # any object can only have one parent, otherwise the chain of applied transforms can become self-inconsistent
        self.obj_to_parents: Dict[int, int] = {}
        # cache the relationship type between two objects (parent,child)
        self.relation_types: Dict[Tuple[int, int], str] = {}

    def add_relation(self, parent: int, child: int, rel_type: str) -> None:
        """
        Add a relationship connection between two objects.

        :param parent: The parent object_id.
        :param child: The child object_id.
        :param rel_type: The type string for the relationship.
        """

        assert parent != child
        if (parent, child) in self.relation_types:
            logger.warn(
                f"Redundant relationship detected. Changing '{parent}' {self.relation_types[(parent, child)]} '{child}' to '{parent}' {rel_type} '{child}'"
            )
        else:
            if parent not in self.obj_to_children:
                self.obj_to_children[parent] = []
            self.obj_to_children[parent].append(child)
            if child in self.obj_to_parents:
                logger.warn(
                    f"Inconsistent relationship requested: child object '{child}' already parented to '{self.obj_to_parents[child]}'. Changing parent to '{parent}' and removing previous relationship."
                )
                self.remove_relation(self.obj_to_parents[child], child)
            self.obj_to_parents[child] = parent
        self.relation_types[(parent, child)] = rel_type

    def remove_relation(self, parent: int, child: int) -> None:
        """
        Remove any relationship between the pair of objects.

        :param parent: The parent object_id.
        :param child: The child object_id.
        """

        assert parent != child
        del self.relation_types[(parent, child)]

        if child in self.obj_to_parents:
            del self.obj_to_parents[child]

        if child in self.obj_to_children[parent]:
            self.obj_to_children[parent].remove(child)
            if len(self.obj_to_children[parent]) == 0:
                del self.obj_to_children[parent]

    def remove_obj_relations(
        self, obj: int, parents_only: bool = False
    ) -> None:
        """
        Remove all relationships for the object.
        Use this to remove an object from the kinematic manager.

        Examples: an object is picked/grasped or object is removed from simulation.

        :param parents_only: If set, remove only the upward relationships (parents) of the object. This maintains child relationships. For example, use this to move a container full of items.
        """

        # NOTE: if any of the below fails, then something went wrong with registration upstream
        if not parents_only and obj in self.obj_to_children:
            for child in self.obj_to_children[obj]:
                self.remove_relation(obj, child)
        if obj in self.obj_to_parents:
            self.remove_relation(self.obj_to_parents[obj], obj)

    def get_root_parents(self) -> List[int]:
        """
        Get a list of root objects: those which are parents but have no parents.

        :return: A list of object_ids.
        """

        return [
            obj_id
            for obj_id in self.obj_to_children
            if obj_id not in self.obj_to_parents
        ]

    def get_human_readable_relationship_forest(
        self, sim: habitat_sim.Simulator, do_print: bool = False
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Get a version of the relationship forest with human readable strings in place of object ids.

        :param sim: We need the Simulator instance to fetch the name strings.
        :param do_print: If true, print the relationship forest nicely in addition to returning it.
        :return: The relationship forest with strings instead of ints. The tuple contains: (object string, relationship type). Note, the strings include both object handles and link names, don't use them to backtrace the objects.
        """

        obj_to_children_strings: Dict[
            str, List[Tuple[str, str]]
        ] = defaultdict(lambda: [])
        # this maps object ids to explainable name strings
        ids_to_obj_names = sutils.get_all_object_ids(sim)
        for parent_id, children in self.obj_to_children.items():
            if len(children) == 0:
                continue
            obj_to_children_strings[ids_to_obj_names[parent_id]] = [
                (
                    ids_to_obj_names[child_id],
                    self.relation_types[(parent_id, child_id)],
                )
                for child_id in children
            ]

        if do_print:
            print("---------------------")
            print("Relationship Forest:")
            for (
                parent_string,
                children_types,
            ) in obj_to_children_strings.items():
                print(f" '{parent_string}': ")
                for child_string, relationship_type in children_types:
                    print(f"   - '{child_string}' | {relationship_type}")

            print("---------------------")

        return obj_to_children_strings


class KinematicRelationshipManager:
    """
    Manages the kinematic relationships between objects such that object states can be manipulated while maintaining said relationships.
    """

    def __init__(self, sim: habitat_sim.Simulator) -> None:
        """..

        :param sim: The Simulator instance to which this KinematicRelationshipManager is attached.
        """
        self.relationship_graph = RelationshipGraph()
        self.sim = sim
        # cache the previous relative transforms for parent->child relationships
        self.prev_snapshot: Dict[int, Dict[int, mn.Matrix4]] = None
        # cache the previous global transforms for the parent objects without parents
        self.prev_root_obj_state: Dict[int, mn.Matrix4] = None
        # note: this must be updated when new AOs are added or objects are removed and object ids could be recycled.
        self.ao_link_map = sutils.get_ao_link_id_map(self.sim)

    def initialize_from_obj_to_rec_pairs(
        self, obj_to_rec: Dict[str, str], receptacles: List[Receptacle]
    ) -> None:
        """
        Initialize the RelationshipGraph from object to receptacle mappings as found in a RearrangeEpisode.

        :param obj_to_rec: Map from object instance names to Receptacle unique names.
        :param receptacles: A list of active Receptacle objects.
        """

        self.relationship_graph = RelationshipGraph()

        # construct a Dict of Receptacle unique_name to Receptacle object
        unique_name_to_rec = {rec.unique_name: rec for rec in receptacles}

        # construct the parent relations
        for obj_handle, rec_unique_name in obj_to_rec.items():
            if rec_unique_name == "floor":
                # NOTE: floor placements are denoted by this explicit name string and do not result in any parenting relationships
                continue

            obj = sutils.get_obj_from_handle(self.sim, obj_handle)
            assert (
                obj is not None
            ), f"Object with handle '{obj_handle}' could not be found in the scene. Has the Episode been initialized?"
            if rec_unique_name not in unique_name_to_rec:
                logger.error(
                    f"Cannot find active receptacle {rec_unique_name}, so cannot create a parent relationship. Skipping. Note that episode is likely invalid."
                )
                continue
            rec = unique_name_to_rec[rec_unique_name]
            parent_obj = sutils.get_obj_from_handle(
                self.sim, rec.parent_object_handle
            )
            parent_id = parent_obj.object_id
            if rec.parent_link is not None and rec.parent_link >= 0:
                # this is a link, get the object id
                link_ids_to_object_ids = dict(
                    (v, k) for k, v in parent_obj.link_object_ids.items()
                )
                parent_id = link_ids_to_object_ids[rec.parent_link]
            self.relationship_graph.add_relation(
                parent_id, obj.object_id, "ontop"
            )

        self.prev_snapshot = self.get_relations_snapshot()
        self.prev_root_obj_state = self.get_root_parents_snapshot()

    def initialize_from_dynamic_ontop(self) -> None:
        """
        Scrape current scene contents to initialize the relationship graph via the "ontop" util, requiring that objects are dynamically simulated such that contact can be used as a heuristic for support relationships.
        """

        self.relationship_graph = RelationshipGraph()
        # do this once now instead of repeating for each ontop
        self.sim.perform_discrete_collision_detection()

        for obj_id in sutils.get_all_object_ids(self.sim).keys():
            parent_obj = sutils.get_obj_from_id(
                self.sim, obj_id, self.ao_link_map
            )
            assert parent_obj is not None, f"Object id {obj_id} is invalid."

            obj_ontop = sutils.ontop(
                self.sim, obj_id, do_collision_detection=False
            )
            for child_id in obj_ontop:
                if child_id == habitat_sim.stage_id:
                    continue
                child_obj = sutils.get_obj_from_id(
                    self.sim, child_id, self.ao_link_map
                )
                if (
                    child_id == child_obj.object_id
                    and child_obj.motion_type
                    != habitat_sim.physics.MotionType.STATIC
                ):
                    # this is a ManagedObject, not a link
                    self.relationship_graph.add_relation(
                        obj_id, child_id, "ontop"
                    )

        self.prev_snapshot = self.get_relations_snapshot()
        self.prev_root_obj_state = self.get_root_parents_snapshot()

    def _get_relations_recursive(
        self, parent_id: int, wip_snapshot: Dict[int, Dict[int, mn.Matrix4]]
    ) -> None:
        """
        Gather the relative transforms for the "work in progress" snapshot relations recursively to all children of the parent_id.

        :param parent_id: The parent of the subtree on which to recurse.
        :param wip_snapshot: The work-in-progress snapshot being constructed by this recursive process. :py:`default_dict(lambda: {})`
        """

        if parent_id not in self.relationship_graph.obj_to_children:
            # no-op for non-parents
            return

        # print(f"_get_relations_recursive ({parent_id} -> {self.relationship_graph.obj_to_children[parent_id]})")
        parent_transform = sutils.get_obj_transform_from_id(
            self.sim, parent_id, self.ao_link_map
        )
        for child_id in self.relationship_graph.obj_to_children[parent_id]:
            # get the relative transform
            child_transform = sutils.get_obj_transform_from_id(
                self.sim, child_id, self.ao_link_map
            )
            relative_transform = parent_transform.inverted() @ child_transform

            # set the transform into the snapshot
            wip_snapshot[parent_id][child_id] = relative_transform

            # apply the operation recursively to each novel child
            if len(wip_snapshot[child_id]) == 0:
                self._get_relations_recursive(child_id, wip_snapshot)

    def get_relations_snapshot(
        self, root_parent_subset: Optional[List[int]] = None
    ) -> Dict[int, Dict[int, mn.Matrix4]]:
        """
        Get the current parent to child transforms for all registered relationships.

        :param root_parent_subset: Optionally, only compute the relations snapshot for a subset of root parents. Default is all root parents.
        :return: A dictionary mapping parent object_id to dictionaries mapping each child object_id to the relative transformation matrix between parent and child.

        NOTE: Some objects may have multiple parents.
        """

        cur_root_parents = self.relationship_graph.get_root_parents()
        root_parents_to_update = []
        if root_parent_subset is None:
            root_parents_to_update = cur_root_parents
        else:
            # screen out potentially stale or inactive parents
            root_parents_to_update = [
                root_parent
                for root_parent in root_parent_subset
                if root_parent in cur_root_parents
            ]
        wip_snapshot: Dict[int, Dict[int, mn.Matrix4]] = defaultdict(
            lambda: {}
        )
        for parent_id in root_parents_to_update:
            self._get_relations_recursive(parent_id, wip_snapshot)

        return wip_snapshot

    def get_root_parents_snapshot(
        self, root_parent_subset: Optional[List[int]] = None
    ) -> Dict[int, mn.Matrix4]:
        """
        Get the global transformations for all root parents: those without any parent.

        :param root_parent_subset: Optionally, only compute the snapshot for a subset of root parents. Default is all root parents.
        :return: dictionary mapping root parent object_ids to their global transformation matrices.
        """

        cur_root_parents = self.relationship_graph.get_root_parents()
        root_parents_to_update = []
        if root_parent_subset is None:
            root_parents_to_update = cur_root_parents
        else:
            # screen out potentially stale or inactive parents
            root_parents_to_update = [
                root_parent
                for root_parent in root_parent_subset
                if root_parent in cur_root_parents
            ]

        root_parent_transforms: Dict[int, mn.Matrix4] = {}
        for obj_id in root_parents_to_update:
            obj_transform = sutils.get_obj_transform_from_id(
                self.sim, obj_id, self.ao_link_map
            )
            assert (
                obj_transform is not None
            ), f"Invalid object id {obj_id} matches no objects or links."
            root_parent_transforms[obj_id] = obj_transform

        return root_parent_transforms

    def _apply_relations_recursive(
        self,
        parent_id: int,
        snapshot: Dict[int, Dict[int, mn.Matrix4]],
        applied_to: Dict[int, bool],
    ) -> None:
        """
        Apply the snapshot transform relations recursively to the subtree with parent_id as root.

        :param parent_id: The parent of the subtree on which to recurse.
        :param snapshot: The snapshot being consumed by this recursive process. Contains both parent->child mapping and transforms.
        :param applied_to: Tracks wether an object has already had a transform applied previously. Prevents infinite recursion in edge cases with cyclic relationships.
        """

        parent_transform = sutils.get_obj_transform_from_id(
            self.sim, parent_id, self.ao_link_map
        )
        for child_id, child_rel_transform in snapshot[parent_id].items():
            # only apply the snapshot to active relationships
            if (
                parent_id,
                child_id,
            ) in self.relationship_graph.relation_types and not applied_to[
                child_id
            ]:
                child_parent_obj = sutils.get_obj_from_id(
                    self.sim, child_id, self.ao_link_map
                )
                if child_parent_obj.object_id == child_id:
                    child_parent_obj.transformation = (
                        parent_transform @ child_rel_transform
                    )
                    applied_to[child_id] = True
                else:
                    # this is an articulated link which is constrained to the parent AO, no action to take here.
                    continue

                # recurse
                self._apply_relations_recursive(child_id, snapshot, applied_to)

    def apply_relationships_snapshot(
        self,
        snapshot: Dict[int, Dict[int, mn.Matrix4]],
        apply_all: bool = True,
    ) -> List[int]:
        """
        Apply all transformations cached in the provided snapshot.

        :param snapshot: The snapshot with parent to child transformations which should be applied.
        :param apply_all: If set, apply all transforms without checking for root parent transform deltas. Use :py:`False` to limit application to dirty transforms.
        :return: The list of root parents for which the subtree transforms were updated.
        """

        # track the root parents for which a recursive update was applied
        updated_root_parents: List[int] = []

        # track applied transforms because some indirect cyclic relationships could result in infinite recursion
        applied_to: Dict[int, bool] = defaultdict(lambda: False)

        # pre-check for parents which should be updated if not apply_all
        apply_transforms_to_subtree: Dict[int, bool] = defaultdict(
            lambda: True
        )
        if not apply_all:
            # check the root parents for state change
            cur_transforms = self.get_root_parents_snapshot()
            for parent_id, cur_transform in cur_transforms.items():
                if (
                    parent_id in self.prev_root_obj_state
                    and self.prev_root_obj_state[parent_id] == cur_transform
                ):
                    apply_transforms_to_subtree[parent_id] = False

        # recursively apply the necessary changes
        for root_parent_id in self.relationship_graph.get_root_parents():
            if apply_transforms_to_subtree[root_parent_id]:
                # print(f"applying to dirty parent {root_parent_id}")
                updated_root_parents.append(root_parent_id)
                self._apply_relations_recursive(
                    root_parent_id, snapshot, applied_to
                )

        return updated_root_parents

    def apply_relations(self) -> List[int]:
        """
        Apply the previous relationship snapshot.
        Call this in the kinematic sim loop when objects are updated.

        :return: The list of root parents for which the subtree transforms were updated.
        """

        return self.apply_relationships_snapshot(
            self.prev_snapshot, apply_all=False
        )

    def update_snapshots(
        self, root_parent_subset: Optional[List[int]] = None
    ) -> None:
        """
        Update the internal previous snapshots.

        :param root_parent_subset: If provided, limit the updates to particular root parents. This is an efficiency option to avoid wasting time in sparse delta situations. For example, when an update is needed after a single state change such as opening or closing a link.
        """

        root_parents_snapshot = self.get_root_parents_snapshot(
            root_parent_subset
        )
        relations_snapshot = self.get_relations_snapshot(root_parent_subset)

        if root_parent_subset is None:
            self.prev_snapshot = relations_snapshot
            self.prev_root_obj_state = root_parents_snapshot
        else:
            # here we need to merge the partial results with the previous snapshot
            current_root_parents = self.relationship_graph.get_root_parents()

            # first add the specified subset
            for root_parent, transform in root_parents_snapshot.items():
                self.prev_root_obj_state[root_parent] = transform
                self.prev_snapshot[root_parent] = relations_snapshot[
                    root_parent
                ]

            # then cull anything stale
            for root_parent_id in list(self.prev_root_obj_state.keys()):
                if root_parent_id not in current_root_parents:
                    del self.prev_root_obj_state[root_parent_id]
                    del self.prev_snapshot[root_parent_id]

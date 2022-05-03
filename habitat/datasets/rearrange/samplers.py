#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import magnum as mn
import numpy as np

import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
from habitat.core.logging import logger
from habitat.datasets.rearrange.receptacle import Receptacle, find_receptacles
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer


class SceneSampler(ABC):
    @abstractmethod
    def num_scenes(self):
        pass

    def reset(self) -> None:
        pass

    @abstractmethod
    def sample(self):
        pass


class SingleSceneSampler(SceneSampler):
    """
    Returns a single provided scene using the sampler API
    """

    def __init__(self, scene: str) -> None:
        self.scene = scene

    def sample(self) -> str:
        return self.scene

    def num_scenes(self) -> int:
        return 1


class MultiSceneSampler(SceneSampler):
    """
    Uniform sampling from a set of scenes.
    """

    def __init__(self, scenes: List[str]) -> None:
        self.scenes = scenes
        assert len(scenes) > 0, "No scenes provided to MultiSceneSampler."

    def sample(self) -> str:
        return self.scenes[random.randrange(0, len(self.scenes))]

    def num_scenes(self) -> int:
        return len(self.scenes)


class ObjectSampler:
    """
    Sample an object from a set and try to place it in the scene from some receptacle set.
    """

    def __init__(
        self,
        object_set: List[str],
        receptacle_sets: List[
            Tuple[List[str], List[str], List[str], List[str]]
        ],
        num_objects: Tuple[int, int] = (1, 1),
        orientation_sample: Optional[str] = None,
        sample_region_ratio: Optional[Dict[str, float]] = None,
        nav_to_min_distance: float = -1.0,
    ) -> None:
        """
        :param nav_to_min_distance: -1.0 means there will be no accessibility constraint. Positive values indicate minimum distance from sampled object to a navigable point.
        """
        self.object_set = object_set
        self.receptacle_sets = receptacle_sets
        self.receptacle_instances: Optional[
            List[Receptacle]
        ] = None  # all receptacles in the scene
        self.receptacle_candidates: Optional[
            List[Receptacle]
        ] = None  # the specific receptacle instances relevant to this sampler
        assert len(self.receptacle_sets) > 0
        self.max_sample_attempts = 1000  # number of distinct object|receptacle pairings to try before giving up
        self.max_placement_attempts = 50  # number of times to attempt a single object|receptacle placement pairing
        self.num_objects = num_objects  # tuple of [min,max] objects to sample
        assert self.num_objects[1] >= self.num_objects[0]
        self.orientation_sample = (
            orientation_sample  # None, "up" (1D), "all" (rand quat)
        )
        if sample_region_ratio is None:
            sample_region_ratio = defaultdict(lambda: 1.0)
        self.sample_region_ratio = sample_region_ratio
        self.nav_to_min_distance = nav_to_min_distance
        self.set_num_samples()
        # More possible parameters of note:
        # - surface vs volume
        # - apply physics stabilization: none, dynamic, projection

    def reset(self) -> None:
        """
        Reset any per-scene variables.
        """
        # receptacle instances should be scraped for every new scene
        self.receptacle_instances = None
        self.receptacle_candidates = None
        # number of objects in the range should be reset each time
        self.set_num_samples()

    def sample_receptacle(
        self,
        sim: habitat_sim.Simulator,
        cull_tilted_receptacles: bool = True,
        tilt_tolerance: float = 0.9,
    ) -> Receptacle:
        """
        Sample a receptacle from the receptacle_set and return relevant information.
        If cull_tilted_receptacles is True, receptacles are culled for objects with local "down" (-Y), not aligned with gravity (unit dot product compared to tilt_tolerance).
        """
        if self.receptacle_instances is None:
            self.receptacle_instances = find_receptacles(sim)

        if self.receptacle_candidates is None:
            self.receptacle_candidates = []
            for receptacle in self.receptacle_instances:
                found_match = False
                for r_set_tuple in self.receptacle_sets:
                    # r_set_tuple = (included_obj_substrs, excluded_obj_substrs, included_receptacle_substrs, excluded_receptacle_substrs)
                    culled = False
                    # first try to cull by exclusion
                    for ex_object_substr in (
                        r_set_tuple[1] and receptacle.parent_object_handle
                    ):
                        if ex_object_substr in receptacle.parent_object_handle:
                            culled = True
                            break
                    for ex_receptacle_substr in r_set_tuple[3]:
                        if ex_receptacle_substr in receptacle.name:
                            culled = True
                            break
                    if culled:
                        break

                    # if the receptacle is stage/global (no object handle) then always a match
                    if receptacle.parent_object_handle is None:
                        # check the inclusion name constraints
                        for name_constraint in r_set_tuple[2]:
                            if name_constraint in receptacle.name:
                                found_match = True
                                break
                        break

                    # then search for inclusion
                    for object_substr in r_set_tuple[0]:
                        if object_substr in receptacle.parent_object_handle:
                            # object substring is valid, try receptacle name constraint
                            for name_constraint in r_set_tuple[2]:
                                if name_constraint in receptacle.name:
                                    # found a valid substring match for this receptacle, stop the search
                                    found_match = True
                                    break
                        if found_match:
                            # break object substr search
                            break
                    if found_match:
                        # break receptacle set search
                        break
                if found_match:
                    # substring match was found, check orientation constraint
                    if cull_tilted_receptacles:
                        obj_down = (
                            receptacle.get_global_transform(sim)
                            .transform_vector(-receptacle.up)
                            .normalized()
                        )
                        gravity_alignment = mn.math.dot(
                            obj_down, sim.get_gravity().normalized()
                        )
                        if gravity_alignment < tilt_tolerance:
                            culled = True
                            logger.info(
                                f"Culled by tilt: '{receptacle.name}', {gravity_alignment}"
                            )
                    if not culled:
                        # found a valid receptacle
                        self.receptacle_candidates.append(receptacle)

        assert (
            len(self.receptacle_candidates) > 0
        ), f"No receptacle instances found matching this sampler's requirements. Likely a sampler config constraint is not feasible for all scenes in the dataset. Cull this scene from your dataset? Scene='{sim.config.sim_cfg.scene_id}'. Receptacle constraints ='{self.receptacle_sets}'"
        target_receptacle = self.receptacle_candidates[
            random.randrange(0, len(self.receptacle_candidates))
        ]
        return target_receptacle

    def sample_object(self) -> str:
        """
        Sample an object handle from the object_set and return it.
        """
        return self.object_set[random.randrange(0, len(self.object_set))]

    def sample_placement(
        self,
        sim: habitat_sim.Simulator,
        object_handle: str,
        receptacle: Receptacle,
        snap_down: bool = False,
        vdb: Optional[DebugVisualizer] = None,
    ) -> Optional[habitat_sim.physics.ManagedRigidObject]:
        """
        Attempt to sample a valid placement of the object in/on a receptacle given an object handle and receptacle information.
        """
        num_placement_tries = 0
        new_object = None
        navmesh_vertices = np.stack(
            sim.pathfinder.build_navmesh_vertices(), axis=0
        )
        # Note: we cache the largest island to reject samples which are primarily accessible from disconnected navmesh regions. This assumption limits sampling to the largest navigable component of any scene.
        self.largest_island_size = max(
            [sim.pathfinder.island_radius(p) for p in navmesh_vertices]
        )

        while num_placement_tries < self.max_placement_attempts:
            num_placement_tries += 1

            # sample the object location
            target_object_position = receptacle.sample_uniform_global(
                sim, self.sample_region_ratio[receptacle.name]
            )

            # instance the new potential object from the handle
            if new_object == None:
                assert sim.get_object_template_manager().get_library_has_handle(
                    object_handle
                ), f"Found no object in the SceneDataset with handle '{object_handle}'."
                new_object = sim.get_rigid_object_manager().add_object_by_template_handle(
                    object_handle
                )

            # try to place the object
            new_object.translation = target_object_position
            if self.orientation_sample is not None:
                if self.orientation_sample == "up":
                    # rotate the object around the gravity direction
                    rot = random.uniform(0, math.pi * 2.0)
                    new_object.rotation = mn.Quaternion.rotation(
                        mn.Rad(rot), mn.Vector3.y_axis()
                    )
                elif self.orientation_sample == "all":
                    # set the object's orientation to a random quaternion
                    new_object.rotation = (
                        habitat_sim.utils.common.random_quaternion()
                    )
            if snap_down:
                support_object_ids = [-1]
                # add support object ids for non-stage receptacles
                if receptacle.is_parent_object_articulated:
                    ao_instance = sim.get_articulated_object_manager().get_object_by_handle(
                        receptacle.parent_object_handle
                    )
                    for (
                        object_id,
                        link_ix,
                    ) in ao_instance.link_object_ids.items():
                        if receptacle.parent_link == link_ix:
                            support_object_ids = [
                                object_id,
                                ao_instance.object_id,
                            ]
                            break
                elif receptacle.parent_object_handle is not None:
                    support_object_ids = [
                        sim.get_rigid_object_manager()
                        .get_object_by_handle(receptacle.parent_object_handle)
                        .object_id
                    ]
                snap_success = sutils.snap_down(
                    sim,
                    new_object,
                    support_object_ids,
                    vdb=vdb,
                )
                if snap_success:
                    logger.info(
                        f"Successfully sampled (snapped) object placement in {num_placement_tries} tries."
                    )
                    if not self._is_accessible(sim, new_object):
                        continue
                    return new_object

            elif not new_object.contact_test():
                logger.info(
                    f"Successfully sampled object placement in {num_placement_tries} tries."
                )
                if not self._is_accessible(sim, new_object):
                    continue
                return new_object

        # if num_placement_tries > self.max_placement_attempts:
        sim.get_rigid_object_manager().remove_object_by_handle(
            new_object.handle
        )
        logger.warning(
            f"Failed to sample {object_handle} placement on {receptacle.name} in {self.max_placement_attempts} tries."
        )

        return None

    def _is_accessible(self, sim, new_object) -> bool:
        """
        Return if the object is within a threshold distance of the nearest
        navigable point and that the nearest navigable point is on the same
        navigation mesh.

        Note that this might not catch all edge cases since the distance is
        based on Euclidean distance. The nearest navigable point may be
        separated from the object by an obstacle.
        """
        if self.nav_to_min_distance == -1:
            return True
        snapped = sim.pathfinder.snap_point(new_object.translation)
        island_radius: float = sim.pathfinder.island_radius(snapped)
        dist = float(
            np.linalg.norm(
                np.array((snapped - new_object.translation))[[0, 2]]
            )
        )
        return (
            dist < self.nav_to_min_distance
            and island_radius == self.largest_island_size
        )

    def single_sample(
        self,
        sim: habitat_sim.Simulator,
        snap_down: bool = False,
        vdb: Optional[DebugVisualizer] = None,
        fixed_target_receptacle=None,
        fixed_obj_handle: Optional[str] = None,
    ) -> Optional[habitat_sim.physics.ManagedRigidObject]:
        # draw a new pairing
        if fixed_obj_handle is None:
            object_handle = self.sample_object()
        else:
            object_handle = fixed_obj_handle
        if fixed_target_receptacle is not None:
            target_receptacle = fixed_target_receptacle
        else:
            target_receptacle = self.sample_receptacle(sim)
        logger.info(
            f"Sampling '{object_handle}' from '{target_receptacle.name}'"
        )

        new_object = self.sample_placement(
            sim, object_handle, target_receptacle, snap_down, vdb
        )

        return new_object, target_receptacle

    def set_num_samples(self):
        self.target_objects_number = (
            random.randrange(self.num_objects[0], self.num_objects[1])
            if self.num_objects[1] > self.num_objects[0]
            else self.num_objects[0]
        )

    def sample(
        self,
        sim: habitat_sim.Simulator,
        target_receptacles,
        snap_down: bool = False,
        vdb: Optional[DebugVisualizer] = None,
    ) -> List[habitat_sim.physics.ManagedRigidObject]:
        """
        Defaults to uniform sample: object -> receptacle -> volume w/ rejection -> repeat.
        Optionally provide a debug visualizer (vdb)
        """
        num_pairing_tries = 0
        new_objects: List[habitat_sim.physics.ManagedRigidObject] = []

        logger.info(
            f"    Trying to sample {self.target_objects_number} from range {self.num_objects}"
        )

        while (
            len(new_objects) < self.target_objects_number
            and num_pairing_tries < self.max_sample_attempts
        ):
            num_pairing_tries += 1
            if len(new_objects) < len(target_receptacles):
                # no objects sampled yet
                new_object, receptacle = self.single_sample(
                    sim, snap_down, vdb, target_receptacles[len(new_objects)]
                )
            else:
                new_object, receptacle = self.single_sample(
                    sim, snap_down, vdb
                )
            if new_object is not None:
                new_objects.append((new_object, receptacle))

        if len(new_objects) >= self.num_objects[0]:
            return new_objects

        # we didn't find the minimum number of placements, so remove all new objects and return
        logger.info(
            f"Failed to sample the minimum number of placements in {self.max_sample_attempts} tries."
        )
        logger.info(
            f"    Only able to sample {len(new_objects)} out of {self.num_objects}..."
        )
        # cleanup
        for new_object in new_objects:
            sim.get_rigid_object_manager().remove_object_by_handle(
                new_object.handle
            )
        return []


class ObjectTargetSampler(ObjectSampler):
    """
    Base sampler for object targets. Instead of sampling from an object template set, sample from an instance set.
    """

    def __init__(
        self,
        object_instance_set: List[habitat_sim.physics.ManagedRigidObject],
        receptacle_sets: List[
            Tuple[List[str], List[str], List[str], List[str]]
        ],
        num_targets: Tuple[int, int] = (1, 1),
        orientation_sample: Optional[str] = None,
        sample_region_ratio: Optional[Dict[str, float]] = None,
        nav_to_min_distance: float = -1.0,
    ) -> None:
        """
        Initialize a standard ObjectSampler but construct the object_set to correspond with specific object instances provided.
        """
        self.object_instance_set = object_instance_set
        object_set = [
            x.creation_attributes.handle for x in self.object_instance_set
        ]
        super().__init__(
            object_set,
            receptacle_sets,
            num_targets,
            orientation_sample,
            sample_region_ratio,
            nav_to_min_distance,
        )

    def sample(
        self,
        sim: habitat_sim.Simulator,
        snap_down: bool = False,
        vdb: Optional[DebugVisualizer] = None,
        target_receptacles=None,
        goal_receptacles=None,
        object_to_containing_receptacle=None,
    ) -> Optional[
        Dict[str, Tuple[habitat_sim.physics.ManagedRigidObject, Receptacle]]
    ]:
        """
        Overridden sampler maps to instances without replacement.
        Returns None if failed, or a dict mapping object handles to new object instances in the sampled target location.
        """

        new_target_objects = {}

        logger.info(
            f"    Trying to sample {self.target_objects_number} targets from range {self.num_objects}"
        )

        if len(target_receptacles) != len(goal_receptacles):
            raise ValueError(
                f"# target receptacles {len(target_receptacles)}, # goal receptacles {len(goal_receptacles)}"
            )
        # The first objects were sampled to be in the target object receptacle
        # locations, so they must be used as the target objects.
        for use_target, use_recep, goal_recep in zip(
            self.object_instance_set, target_receptacles, goal_receptacles
        ):
            if object_to_containing_receptacle[use_target.handle] != use_recep:
                raise ValueError(
                    f"Object {use_target.handle}, contained {object_to_containing_receptacle[use_target.handle].name}, target receptacle {use_recep.name}"
                )
            new_object, receptacle = self.single_sample(
                sim,
                snap_down,
                vdb,
                goal_recep,
                use_target.creation_attributes.handle,
            )
            if new_object is None:
                break
            new_target_objects[use_target.handle] = (
                new_object,
                use_recep,
            )

        # Did we successfully place all the objects?
        if len(new_target_objects) == self.target_objects_number:
            return new_target_objects

        # we didn't find all placements, so remove all new objects and return
        logger.info(
            f"Failed to sample all target placements in {self.max_sample_attempts} tries."
        )
        logger.info(
            f"    Only able to sample {len(new_target_objects)} targets out of {len(self.object_instance_set)}..."
        )
        # cleanup
        for new_object, _ in new_target_objects.values():
            sim.get_rigid_object_manager().remove_object_by_handle(
                new_object.handle
            )
        return None


class ArticulatedObjectStateSampler:
    def __init__(
        self, ao_handle: str, link_name: str, state_range: Tuple[float, float]
    ) -> None:
        self.ao_handle = ao_handle
        self.link_name = link_name
        self.state_range = state_range
        assert self.state_range[1] >= self.state_range[0]

    def sample(
        self, sim: habitat_sim.Simulator, receptacles=None
    ) -> Optional[
        Dict[habitat_sim.physics.ManagedArticulatedObject, Dict[int, float]]
    ]:
        """
        For all matching AOs in the scene, sample and apply the joint state for this sampler.
        Return a list of tuples (instance_handle, link_name, state)
        """
        ao_states: Dict[
            habitat_sim.physics.ManagedArticulatedObject, Dict[int, float]
        ] = {}
        # TODO: handle sampled invalid states (e.g. fridge open into wall in some scenes)
        aom = sim.get_articulated_object_manager()
        # get all AOs in the scene with the configured handle as a substring
        matching_ao_instances = aom.get_objects_by_handle_substring(
            self.ao_handle
        ).values()
        for ao_instance in matching_ao_instances:
            # now find a matching link
            for link_ix in range(ao_instance.num_links):
                if ao_instance.get_link_name(link_ix) == self.link_name:
                    # found a matching link, sample the state
                    joint_state = random.uniform(
                        self.state_range[0], self.state_range[1]
                    )
                    # set the joint state
                    pose = ao_instance.joint_positions
                    pose[
                        ao_instance.get_link_joint_pos_offset(link_ix)
                    ] = joint_state
                    ao_instance.joint_positions = pose
                    if ao_instance not in ao_states:
                        ao_states[ao_instance] = {}
                    ao_states[ao_instance][link_ix] = joint_state
                    break
        return ao_states


class CompositeArticulatedObjectStateSampler(ArticulatedObjectStateSampler):
    """
    Samples multiple articulated states simultaneously with rejection of invalid configurations.
    """

    def __init__(
        self,
        ao_sampler_params: Dict[str, Dict[str, Tuple[float, float, bool]]],
    ) -> None:
        """
        ao_sampler_params : {ao_handle -> {link_name -> (min, max)}}
        """
        self.ao_sampler_params = ao_sampler_params
        self.max_iterations = 50
        # validate the ranges
        for ao_handle in ao_sampler_params:
            for link_name in ao_sampler_params[ao_handle]:
                assert (
                    ao_sampler_params[ao_handle][link_name][1]
                    >= ao_sampler_params[ao_handle][link_name][0]
                )

    def sample(
        self, sim: habitat_sim.Simulator, receptacles: List[Receptacle]
    ) -> Optional[
        Dict[habitat_sim.physics.ManagedArticulatedObject, Dict[int, float]]
    ]:
        """
        Iterative rejection sampling of all joint states specified in parameters.
        Return a list of tuples (instance_handle, link_name, state)
        On failure, return None.
        """
        ids_to_names = sutils.get_all_object_ids(sim)
        ids_to_names[-1] = "_stage"
        logger.info(ids_to_names)
        # first collect all instances associated with requested samplers
        aom = sim.get_articulated_object_manager()
        matching_ao_instances: Dict[
            str, List[habitat_sim.physics.ManagedArticulatedObject]
        ] = {}
        for ao_handle in self.ao_sampler_params:
            matching_ao_instances[
                ao_handle
            ] = aom.get_objects_by_handle_substring(ao_handle).values()

        # construct an efficiently iterable structure for reject sampling of link states
        link_sample_params: Dict[
            habitat_sim.physics.ManagedArticulatedObject,
            Dict[int, Tuple[float, float, bool]],
        ] = {}
        for ao_handle, ao_instances in matching_ao_instances.items():
            for ao_instance in ao_instances:
                for link_ix in range(ao_instance.num_links):
                    link_name = ao_instance.get_link_name(link_ix)
                    if link_name in self.ao_sampler_params[ao_handle]:
                        if ao_instance not in link_sample_params:
                            link_sample_params[ao_instance] = {}
                        assert (
                            link_ix not in link_sample_params[ao_instance]
                        ), f"Joint sampler configuration creating duplicate sampler requests for object '{ao_handle}', instance '{ao_instance.handle}', link {link_name}."
                        link_sample_params[ao_instance][
                            link_ix
                        ] = self.ao_sampler_params[ao_handle][link_name]

        for _iteration in range(self.max_iterations):
            ao_states: Dict[
                habitat_sim.physics.ManagedArticulatedObject, Dict[int, float]
            ] = {}
            # sample a composite joint configuration
            for ao_instance, link_ranges in link_sample_params.items():
                ao_states[ao_instance] = {}
                # NOTE: only query and set pose once per instance for efficiency
                pose = ao_instance.joint_positions
                for link_ix, joint_range in link_ranges.items():
                    should_sample_all_joints = joint_range[2]
                    matching_recep = None
                    for recep in receptacles:
                        link_matches = (
                            link_ix == recep.parent_link
                        ) or should_sample_all_joints
                        if (
                            ao_instance.handle == recep.parent_object_handle
                            and link_matches
                        ):
                            matching_recep = recep
                            break

                    if matching_recep is not None:
                        # If this is true, this means that the receptacle AO must be opened. That is because
                        # the object is spawned inside the fridge OR inside the kitchen counter BUT not on top of the counter
                        # because in this case all drawers must be closed.
                        # TODO: move this receptacle access logic to the ao_config files in a future refactor
                        joint_state = random.uniform(
                            joint_range[0], joint_range[1]
                        )
                    else:
                        joint_state = pose[
                            ao_instance.get_link_joint_pos_offset(link_ix)
                        ]
                    pose[
                        ao_instance.get_link_joint_pos_offset(link_ix)
                    ] = joint_state
                    ao_states[ao_instance][link_ix] = joint_state
                ao_instance.joint_positions = pose

            # validate the new configuration (contact check every instance)
            valid_configuration = True
            for ao_handle in matching_ao_instances:
                for ao_instance in matching_ao_instances[ao_handle]:
                    if ao_instance.contact_test():
                        logger.info(
                            f"ao_handle = {ao_handle} failed contact test."
                        )
                        sim.perform_discrete_collision_detection()
                        cps = sim.get_physics_contact_points()
                        logger.info(ao_instance.handle)
                        for cp in cps:
                            if (
                                ao_instance.handle
                                in ids_to_names[cp.object_id_a]
                                or ao_instance.handle
                                in ids_to_names[cp.object_id_b]
                            ):
                                logger.info(
                                    f" contact between ({cp.object_id_a})'{ids_to_names[cp.object_id_a]}' and ({cp.object_id_b})'{ids_to_names[cp.object_id_b]}'"
                                )
                        valid_configuration = False
                        break
                if not valid_configuration:
                    break

            if valid_configuration:
                # success
                return ao_states

        # failed to find a valid configuration
        return None

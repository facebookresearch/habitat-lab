#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import magnum as mn

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
        sample_region_ratio: float = 1.0,
    ) -> None:
        self.object_set = object_set
        self.receptacle_sets = receptacle_sets
        self.receptacle_instances: Optional[
            List[Receptacle]
        ] = None  # all receptacles in the scene
        self.receptacle_candidates: Optional[
            List[Receptacle]
        ] = None  # the specific receptacle instances relevant to this sampler
        assert len(self.object_set) > 0
        assert len(self.receptacle_sets) > 0
        self.max_sample_attempts = 1000  # number of distinct object|receptacle pairings to try before giving up
        self.max_placement_attempts = 50  # number of times to attempt a single object|receptacle placement pairing
        self.num_objects = num_objects  # tuple of [min,max] objects to sample
        assert self.num_objects[1] >= self.num_objects[0]
        self.orientation_sample = (
            orientation_sample  # None, "up" (1D), "all" (rand quat)
        )
        self.sample_region_ratio = sample_region_ratio
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
        while num_placement_tries < self.max_placement_attempts:
            num_placement_tries += 1

            # sample the object location
            target_object_position = receptacle.sample_uniform_global(
                sim, self.sample_region_ratio
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
                    return new_object

            elif not new_object.contact_test():
                logger.info(
                    f"Successfully sampled object placement in {num_placement_tries} tries."
                )
                return new_object

        # if num_placement_tries > self.max_placement_attempts:
        sim.get_rigid_object_manager().remove_object_by_handle(
            new_object.handle
        )
        logger.info(
            f"Failed to sample object placement in {self.max_placement_attempts} tries."
        )
        return None

    def single_sample(
        self,
        sim: habitat_sim.Simulator,
        snap_down: bool = False,
        vdb: Optional[DebugVisualizer] = None,
    ) -> Optional[habitat_sim.physics.ManagedRigidObject]:
        # draw a new pairing
        object_handle = self.sample_object()
        target_receptacle = self.sample_receptacle(sim)
        logger.info(
            f"Sampling '{object_handle}' from '{target_receptacle.name}'"
        )

        new_object = self.sample_placement(
            sim, object_handle, target_receptacle, snap_down, vdb
        )

        return new_object

    def sample(
        self,
        sim: habitat_sim.Simulator,
        snap_down: bool = False,
        vdb: Optional[DebugVisualizer] = None,
    ) -> List[habitat_sim.physics.ManagedRigidObject]:
        """
        Defaults to uniform sample: object -> receptacle -> volume w/ rejection -> repeat.
        Optionally provide a debug visualizer (vdb)
        """
        num_pairing_tries = 0
        new_objects: List[habitat_sim.physics.ManagedRigidObject] = []

        target_objects_number = (
            random.randrange(self.num_objects[0], self.num_objects[1])
            if self.num_objects[1] > self.num_objects[0]
            else self.num_objects[0]
        )
        logger.info(
            f"    Trying to sample {target_objects_number} from range {self.num_objects}"
        )

        while (
            len(new_objects) < target_objects_number
            and num_pairing_tries < self.max_sample_attempts
        ):
            num_pairing_tries += 1
            new_object = self.single_sample(sim, snap_down, vdb)
            if new_object is not None:
                new_objects.append(new_object)

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
    ) -> None:
        """
        Initialize a standard ObjectSampler but construct the object_set to correspond with specific object instances provided.
        """
        self.object_instance_set = object_instance_set
        object_set = [
            x.creation_attributes.handle for x in self.object_instance_set
        ]
        super().__init__(
            object_set, receptacle_sets, num_targets, orientation_sample
        )

    def sample(
        self,
        sim: habitat_sim.Simulator,
        snap_down: bool = False,
        vdb: Optional[DebugVisualizer] = None,
    ) -> Optional[Dict[str, habitat_sim.physics.ManagedRigidObject]]:
        """
        Overridden sampler maps to instances without replacement.
        Returns None if failed, or a dict mapping object handles to new object instances in the sampled target location.
        """

        # initialize all targets to None
        targets_found = 0
        new_target_objects = {}

        target_number = (
            random.randrange(self.num_objects[0], self.num_objects[1])
            if self.num_objects[1] > self.num_objects[0]
            else self.num_objects[0]
        )
        target_number = min(target_number, len(self.object_instance_set))
        logger.info(
            f"    Trying to sample {target_number} targets from range {self.num_objects}"
        )

        num_pairing_tries = 0
        while (
            targets_found < target_number
            and num_pairing_tries < self.max_sample_attempts
        ):
            num_pairing_tries += 1
            new_object = self.single_sample(sim, snap_down, vdb)
            if new_object is not None:
                targets_found += 1
                found_match = False
                for object_instance in self.object_instance_set:
                    if (
                        object_instance.creation_attributes.handle
                        == new_object.creation_attributes.handle
                        and object_instance.handle not in new_target_objects
                    ):
                        new_target_objects[object_instance.handle] = new_object
                        found_match = True
                        # remove this object instance match from future pairings
                        self.object_set.remove(
                            new_object.creation_attributes.handle
                        )
                        break
                assert (
                    found_match is True
                ), "Failed to match instance to generated object. Shouldn't happen, must be a bug."

        if len(new_target_objects) >= self.num_objects[0]:
            return new_target_objects

        # we didn't find all placements, so remove all new objects and return
        logger.info(
            f"Failed to sample all target placements in {self.max_sample_attempts} tries."
        )
        logger.info(
            f"    Only able to sample {len(new_target_objects)} targets out of {len(self.object_instance_set)}..."
        )
        # cleanup
        for new_object in new_target_objects.values():
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
        self, sim: habitat_sim.Simulator
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
        for ao_instance in matching_ao_instances.values():
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
        self, ao_sampler_params: Dict[str, Dict[str, Tuple[float, float]]]
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
        self, sim: habitat_sim.Simulator
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
            Dict[int, Tuple[float, float]],
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
                    joint_state = random.uniform(
                        joint_range[0], joint_range[1]
                    )
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

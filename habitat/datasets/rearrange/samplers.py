#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Set
from habitat_sim.simulator import Simulator

import magnum as mn
import numpy as np
from matplotlib.path import Path

import habitat.datasets.rearrange.sim_utilities as sutils
import habitat_sim


class SceneSampler(ABC):
    @abstractmethod
    def num_scenes(self):
        pass

    def reset(self)-> None:
        pass

    @abstractmethod
    def sample(self):
        pass


class SingleSceneSampler(SceneSampler):
    """
    Returns a single provided scene using the sampler API
    """

    def __init__(self, scene:str)-> None:
        self.scene = scene

    def sample(self)-> str:
        return self.scene

    def num_scenes(self)-> int:
        return 1


class MultiSceneSampler(SceneSampler):
    """
    Uniform sampling from a set of scenes.
    """

    def __init__(self, scenes:List[str])-> None:
        self.scenes = scenes

    def sample(self)-> str:
        return self.scenes[random.randrange(0, len(self.scenes))]

    def num_scenes(self)-> int:
        return len(self.scenes)


class SceneSubsetSampler(SceneSampler):
    """
    Uniform sampling from a set of scenes. Requires an initialized Simulator with a loaded SceneDataset before first sample query to construct the
    """

    def __init__(self, included_scene_subsets:Set[str], excluded_scene_subsets:Set[str], sim:habitat_sim.Simulator)-> None:
        self.scenes:List[str] = []
        # NOTE: each scene subset is a partial string handle, so use substring search to find all matches
        all_scene_handles = sim.metadata_mediator.get_scene_handles()
        for scene_handle in all_scene_handles:
            excluded = False
            for excluded_substring in excluded_scene_subsets:
                if excluded_substring in scene_handle:
                    excluded = True
                    break
            if not excluded:
                for included_substring in included_scene_subsets:
                    if included_substring in scene_handle:
                        self.scenes.append(scene_handle)
                        break
        # remove any duplicates
        self.scenes = list(set(self.scenes))

    def sample(self)-> str:
        assert (
            len(self.scenes) > 0
        ), "SceneSubsetSampler.sample() Error: No scenes to sample."
        return self.scenes[random.randrange(0, len(self.scenes))]

    def num_scenes(self)-> int:
        if len(self.scenes) == 0:
            print(
                "SceneSubsetSampler: scene set is empty. Sampler may not be initialized."
            )
        return len(self.scenes)


class ObjectSampler:
    """
    Sample an object from a set and try to place it in the scene from some receptacle set.
    """

    def __init__(
        self,
        object_set:List[str],
        receptacle_set:List[str],
        num_objects:Tuple[int, int]=(1, 1),
        orientation_sample:Optional[str]=None,
    )-> None:
        self.object_set = object_set
        self.receptacle_set = receptacle_set
        self.receptacle_instances:Optional[List[sutils.Receptacle]] = None
        assert len(self.object_set) > 0
        assert len(self.receptacle_set) > 0
        self.max_sample_attempts = 1000  # number of distinct object|receptacle pairings to try before giving up
        self.max_placement_attempts = 50  # number of times to attempt a single object|receptacle placement pairing
        self.num_objects = num_objects  # tuple of [min,max] objects to sample
        assert self.num_objects[1] >= self.num_objects[0]
        self.orientation_sample = (
            orientation_sample  # None, "up" (1D), "all" (rand quat)
        )
        # More possible parameters of note:
        # - surface vs volume
        # - apply physics stabilization: none, dynamic, projection

    def reset(self)-> None:
        """
        Reset any per-scene variables.
        """
        # instances should be scraped for every new scene
        self.receptacle_instances = None

    def sample_receptacle(self, sim:habitat_sim.Simulator)->sutils.Receptacle:
        """
        Sample a receptacle from the receptacle_set and return relevant information.
        """
        if self.receptacle_instances is None:
            self.receptacle_instances = sutils.find_receptacles(sim)

        valid_receptacle_targets = []
        receptacle_candidates = self.receptacle_set[:]  # copy

        while len(receptacle_candidates) > 0:
            # uniformly sample a receptacle handle
            receptacle_handle = receptacle_candidates[
                random.randrange(0, len(receptacle_candidates))
            ]

            # get all receptacles matching the handle and filter them by constraints
            for receptacle in self.receptacle_instances:
                if receptacle_handle in receptacle.parent_object_handle:
                    valid_receptacle_targets.append(receptacle)

            if len(valid_receptacle_targets) == 0:
                # we didn't find any matching instances, so remove this receptacle from consideration
                receptacle_candidates.remove(receptacle_handle)
            else:
                break
        assert (
            len(valid_receptacle_targets) > 0
        ), f"No receptacle instances found matching this sampler. Likely an EpisodeGenerator config problem. Cull this scene? Scene='{sim.config.sim_config.scene_id}'. Receptacles='{self.receptacle_set}'"
        target_receptacle = valid_receptacle_targets[
            random.randrange(0, len(valid_receptacle_targets))
        ]
        return target_receptacle

    def sample_object(self)->str:
        """
        Sample an object from the object_set and return its handle.
        """
        return self.object_set[random.randrange(0, len(self.object_set))]

    def sample_placement(
        self, sim:habitat_sim.Simulator, object_handle:str, receptacle:sutils.Receptacle, snap_down:bool=False, vdb:Optional[sutils.DebugVisualizer]=None
    )->Optional[habitat_sim.physics.ManagedRigidObject]:
        """
        Attempt to sample a valid placement of the object in/on a receptacle given an object handle and receptacle information.
        """
        num_placement_tries = 0
        new_object = None
        while num_placement_tries < self.max_placement_attempts:
            num_placement_tries += 1

            # sample the object location
            target_object_position = receptacle.sample_uniform_global(sim)

            # try to place the object
            if new_object == None:
                # find the full object handle and ensure uniqueness
                matching_object_handles = (
                    sim.get_object_template_manager().get_template_handles(
                        object_handle
                    )
                )
                assert len(matching_object_handles) == 1
                new_object = sim.get_rigid_object_manager().add_object_by_template_handle(
                    matching_object_handles[0]
                )
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
                if receptacle.is_parent_object_articulated:
                    # TODO: flesh this out
                    pass
                else:
                    snap_success = sutils.snap_down(
                        sim,
                        new_object,
                        sim.get_rigid_object_manager().get_object_by_handle(
                            receptacle.parent_object_handle
                        ),
                        vdb=vdb,
                    )
                    if snap_success:
                        print(
                            f"Successfully sampled (snapped) object placement in {num_placement_tries} tries."
                        )
                        return new_object

            elif not new_object.contact_test():
                print(
                    f"Successfully sampled object placement in {num_placement_tries} tries."
                )
                return new_object

        # if num_placement_tries > self.max_placement_attempts:
        sim.get_rigid_object_manager().remove_object_by_handle(
            new_object.handle
        )
        print(
            f"Failed to sample object placement in {self.max_placement_attempts} tries."
        )
        return None

    def single_sample(self, sim:habitat_sim.Simulator, snap_down:bool=False, vdb:Optional[sutils.DebugVisualizer]=None)->Optional[habitat_sim.physics.ManagedRigidObject]:
        # draw a new pairing
        object_handle = self.sample_object()
        target_receptacle = self.sample_receptacle(sim)
        print(f" target_receptacle = {target_receptacle}")
        print(f"Sampling {object_handle} from {target_receptacle.name}")

        new_object = self.sample_placement(
            sim, object_handle, target_receptacle, snap_down, vdb
        )

        return new_object

    def sample(self, sim:habitat_sim.Simulator, snap_down:bool=False, vdb:Optional[sutils.DebugVisualizer]=None)-> List[habitat_sim.physics.ManagedRigidObject]:
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
        print(
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
        print(
            f"Failed to sample the minimum number of placements in {self.max_sample_attempts} tries."
        )
        print(
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
        object_instance_set:List[habitat_sim.physics.ManagedRigidObject],
        receptacle_set:List[str],
        num_targets:Tuple[int, int]=(1, 1),
        orientation_sample:Optional[str]=None,
    )-> None:
        """
        Initialize a standard ObjectSampler but construct the object_set to correspond with specific object instances provided.
        """
        self.object_instance_set = object_instance_set
        object_set = [
            x.creation_attributes.handle for x in self.object_instance_set
        ]
        super().__init__(
            object_set, receptacle_set, num_targets, orientation_sample
        )

    def sample(self, sim:habitat_sim.Simulator, snap_down:bool=False, vdb:Optional[sutils.DebugVisualizer]=None)-> Optional[Dict[str, habitat_sim.physics.ManagedRigidObject]]:
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
        print(
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
                    ):
                        if not object_instance.handle in new_target_objects:
                            new_target_objects[
                                object_instance.handle
                            ] = new_object
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
        print(
            f"Failed to sample all target placements in {self.max_sample_attempts} tries."
        )
        print(
            f"    Only able to sample {len(new_target_objects)} targets out of {len(self.object_instance_set)}..."
        )
        # cleanup
        for new_object in new_target_objects.values():
            sim.get_rigid_object_manager().remove_object_by_handle(
                new_object.handle
            )
        return None


class ArticulatedObjectStateSampler:
    def __init__(self, ao_handle:str, link_name:str, state_range:Tuple[float, float])-> None:
        self.ao_handle = ao_handle
        self.link_name = link_name
        self.state_range = state_range
        assert self.state_range[1] >= self.state_range[0]

    def sample(self, sim:habitat_sim.Simulator)-> List[Tuple[str, str, float]]:
        """
        For all matching AOs in the scene, sample and apply the joint state for this sampler.
        Return a list of tuples (instance_handle, link_name, state)
        """
        ao_states = []
        # TODO: handle sampled invalid states (e.g. fridge open into wall in some scenes)
        aom = sim.get_articulated_object_manager()
        # get all AOs in the scene with the configured handle as a substring
        matching_ao_instances = aom.get_objects_by_handle_substring(
            self.ao_handle
        )
        for ao_handle, ao_instance in matching_ao_instances.items():
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
                    ao_states.append((ao_handle, self.link_name, joint_state))
                    break
        return ao_states

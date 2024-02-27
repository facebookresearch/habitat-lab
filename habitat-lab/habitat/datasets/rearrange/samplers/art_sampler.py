#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Dict, List, Optional, Tuple

import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
from habitat.core.logging import logger
from habitat.datasets.rearrange.samplers.receptacle import Receptacle


class ArticulatedObjectStateSampler:
    def __init__(
        self, ao_handle: str, link_name: str, state_range: Tuple[float, float]
    ) -> None:
        self.ao_handle = ao_handle
        self.link_name = link_name
        self.state_range = state_range
        assert self.state_range[1] >= self.state_range[0]

    def _sample_joint_state(self):
        return random.uniform(self.state_range[0], self.state_range[1])

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
                    joint_state = self._sample_joint_state()
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


class ArtObjCatStateSampler(ArticulatedObjectStateSampler):
    def __init__(
        self, ao_handle: str, link_name: str, state_range: Tuple[float, float]
    ):
        super().__init__(ao_handle, link_name, state_range)

    def _sample_joint_state(self):
        return random.choice(self.state_range)


class CompositeArticulatedObjectStateSampler(ArticulatedObjectStateSampler):
    """
    Samples multiple articulated states simultaneously with rejection of invalid configurations.
    """

    def __init__(
        self,
        ao_sampler_params: Dict[str, Dict[str, Tuple[float, float, bool]]],
        apply_prob: Optional[float],
    ) -> None:
        """
        ao_sampler_params : {ao_handle -> {link_name -> (min, max)}}
        """
        self.ao_sampler_params = ao_sampler_params
        self.max_iterations = 50
        self._apply_prob = apply_prob
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
        ids_to_names[habitat_sim.stage_id] = "_stage"
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

                    should_sample = True
                    if self._apply_prob is not None:
                        should_sample = self._apply_prob < random.random()

                    if matching_recep is not None and should_sample:
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

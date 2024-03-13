#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import habitat_sim
from habitat.core.logging import logger
from habitat.datasets.rearrange.samplers.object_sampler import ObjectSampler
from habitat.datasets.rearrange.samplers.receptacle import (
    Receptacle,
    ReceptacleTracker,
)
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer


class ObjectTargetSampler(ObjectSampler):
    """
    Base sampler for object targets. Instead of sampling from an object template set, sample from an instance set.
    """

    def __init__(
        self,
        object_instance_set: List[habitat_sim.physics.ManagedRigidObject],
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize a standard ObjectSampler but construct the object_set to correspond with specific object instances provided.
        """

        self.object_instance_set = object_instance_set
        object_set = [
            x.creation_attributes.handle for x in self.object_instance_set
        ]
        super().__init__(object_set, *args, **kwargs)

    def sample(
        self,
        sim: habitat_sim.Simulator,
        recep_tracker: ReceptacleTracker,
        snap_down: bool = False,
        dbv: Optional[DebugVisualizer] = None,
        target_receptacles=None,
        goal_receptacles=None,
        object_to_containing_receptacle=None,
    ) -> Optional[
        Dict[str, Tuple[habitat_sim.physics.ManagedRigidObject, Receptacle]]
    ]:
        """
        Overridden sampler maps to instances without replacement.

        :param sim: The Simulator instance.
        :param recep_tracker: The ReceptacleTracker containing ReceptacleSet and use information.
        :param snap_down: Whether or not to use the snapdown utility for placement.
        :param dbv: An optional DebugVisualizer (dbv) to gather placement debug images.
        :param target_receptacles: Specify precise Receptacles to use instead of sampling.
        :param goal_receptacles: Provide the list of Receptacles pre-selected for goal placement.
        :param object_to_containing_receptacle: Dictionary mapping object handles to receptacles containing them.

        :return: None if failed. Otherwise a dict mapping object handles to new object instances in the sampled target location.
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
                recep_tracker,
                snap_down,
                dbv,
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

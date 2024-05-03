#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Set

from habitat.sims.habitat_simulator import sim_utilities
from habitat.tasks.rearrange.articulated_agent_manager import (
    ArticulatedAgentManager,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat_sim.physics import ManagedArticulatedObject


class World:
    """
    Global world state shared by each user.
    """

    def __init__(
        self,
        sim: RearrangeSim,
    ):
        self._sim = sim

        # Cache of all held objects.
        # Prevents users from picking objects held by others.
        self._all_held_object_ids: Set[int] = set()
        # Cache of all opened articulated object links.
        self._opened_link_set: Set = set()
        # Cache of all link IDs and their parent articulated objects.
        # Used to speed-up sim queries.
        self._link_id_to_ao_map: Dict[int, int] = {}
        # Cache of pickable objects IDs.
        self._pickable_object_ids: Set[int] = set()
        # Cache of interactable objects IDs.
        self._interactable_object_ids: Set[int] = set()
        # Cache of agent articulated object IDs.
        self._agent_object_ids: Set[int] = set()

    def reset(self) -> None:
        """
        Reset the world state. Call every time the scene changes.
        """
        sim = self._sim
        self._all_held_object_ids = set()
        self._opened_link_set = set()
        self._link_id_to_ao_map = sim_utilities.get_ao_link_id_map(sim)

        # Find pickable objects.
        self._pickable_object_ids = set(sim._scene_obj_ids)
        for pickable_obj_id in self._pickable_object_ids:
            rigid_obj = self.get_rigid_object(pickable_obj_id)
            # Ensure that rigid objects are collidable.
            rigid_obj.collidable = True

        # Get set of interactable articulated object links.
        # Exclude all agents.
        agent_articulated_objects = set()
        agent_manager: ArticulatedAgentManager = sim.agents_mgr
        for agent_index in range(len(agent_manager)):
            agent = agent_manager[agent_index]
            agent_ao = agent.articulated_agent.sim_obj
            agent_articulated_objects.add(agent_ao.object_id)
            self._agent_object_ids.add(agent_ao.object_id)
        self._interactable_object_ids = set()
        aom = sim.get_articulated_object_manager()
        all_ao: List[
            ManagedArticulatedObject
        ] = aom.get_objects_by_handle_substring().values()
        # Classify all non-root links.
        for ao in all_ao:
            for link_object_id in ao.link_object_ids:
                if link_object_id != ao.object_id:
                    # Link is part of an agent.
                    if ao.object_id in agent_articulated_objects:
                        self._agent_object_ids.add(link_object_id)
                    # Link is not part of an agent.
                    else:
                        self._interactable_object_ids.add(link_object_id)
                        # Make sure that link is in "closed" state.
                        link_index = self.get_link_index(link_object_id)
                        if link_index:
                            sim_utilities.close_link(ao, link_index)

    def get_rigid_object(self, object_id: int) -> Optional[Any]:
        """Get the rigid object with the specified ID. Returns None if unsuccessful."""
        rom = self._sim.get_rigid_object_manager()
        return rom.get_object_by_id(object_id)

    def get_articulated_object(self, object_id: int) -> Optional[Any]:
        """Get the articulated object with the specified ID. Returns None if unsuccessful."""
        aom = self._sim.get_articulated_object_manager()
        return aom.get_object_by_id(object_id)

    def get_link_index(self, object_id: int) -> int:
        """Get the index of a link. Returns None if unsuccessful."""
        obj = sim_utilities.get_obj_from_id(
            self._sim, object_id, self._link_id_to_ao_map
        )
        if (
            obj is not None
            and isinstance(obj, ManagedArticulatedObject)
            and object_id in obj.link_object_ids
        ):
            return obj.link_object_ids[object_id]
        return None

    def get_agent_object_ids(self, agent_index: int) -> Set[int]:
        """Get the IDs of objects composing an agent (including links)."""
        # TODO: Cache
        sim = self._sim
        agent_manager: ArticulatedAgentManager = sim.agents_mgr
        agent_object_ids: Set[int] = set()
        agent = agent_manager[agent_index]
        agent_ao = agent.articulated_agent.sim_obj
        agent_object_ids.add(agent_ao.object_id)

        for link_object_id in agent_ao.link_object_ids:
            agent_object_ids.add(link_object_id)

        return agent_object_ids

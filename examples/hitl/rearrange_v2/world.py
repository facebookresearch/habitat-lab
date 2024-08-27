#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union

from omegaconf import DictConfig

from habitat.sims.habitat_simulator import sim_utilities
from habitat.sims.habitat_simulator.object_state_machine import (
    ObjectStateMachine,
    ObjectStateSpec,
)
from habitat.tasks.rearrange.articulated_agent_manager import (
    ArticulatedAgentManager,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat_sim.physics import (
    ManagedBulletArticulatedObject,
    ManagedBulletRigidObject,
)
from habitat_sim.scene import SemanticRegion


@dataclass
class ObjectStateInfo:
    state_spec: ObjectStateSpec
    value: Any


class World:
    """
    Global world state shared by each user.
    Encapsulates all information available in a scene.
    """

    def __init__(
        self,
        sim: RearrangeSim,
        config: DictConfig,
    ):
        self._sim = sim
        self._config = config

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
        # Cache of region names.
        self._regions: Dict[int, SemanticRegion] = {}
        # Cache of all active states.
        self._all_states: Optional[Dict[str, ObjectStateSpec]] = None
        # Cache of object states.
        self._state_snapshot_dict: Dict[str, Dict[str, Any]] = {}
        # Object state container.
        self._object_state_machine: Optional[ObjectStateMachine] = None
        # Cache of categories for each object handles.
        self._object_handle_categories: Dict[str, Optional[str]] = {}
        # Interface for handling scene metadata.
        self._metadata_interface: Optional[Any] = None

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

        # Find regions.
        regions = sim.semantic_scene.regions
        for region_index in range(len(regions)):
            self._regions[region_index] = regions[region_index]

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
            ManagedBulletArticulatedObject
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

        # Object id <-> handle mapping
        self._id_to_handle: Dict[int, str] = {}
        self._handle_to_id: Dict[str, int] = {}
        all_objects = sim_utilities.get_all_objects(sim)
        for obj in all_objects:
            self._id_to_handle[obj.object_id] = obj.handle
            self._handle_to_id[obj.handle] = obj.object_id

        # Try initializing object state and metadata information.
        self._try_initialize_object_states(sim, self._config)

    def update(self, dt: float) -> None:
        """
        Update the world state. Call every frame.
        """
        sim = self._sim
        osm = self._object_state_machine

        if osm is not None:
            osm.update_states(sim, dt)
            self._state_snapshot_dict = osm.get_snapshot_dict(sim)

    def get_state_snapshot_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a snapshot of the current state of all objects.

        Example:
        {
            "is_powered_on": {
                "my_lamp.0001": True,
                "my_oven": False,
                ...
            },
            "is_clean": {
                "my_dish.0002:" False,
                ...
            },
            ...
        }
        """
        return self._state_snapshot_dict

    def get_all_states(self) -> Dict[str, ObjectStateSpec]:
        """
        Get a map of all object states active in this world.
        The key is the state name, and the value is the state specification.
        """
        return self._all_states

    def get_states_for_object_handle(
        self, object_handle: str
    ) -> List[ObjectStateInfo]:
        """
        Get a list of all object states for a given object handle.
        """
        object_states: Dict[str, Any] = {}
        state_snapshot = self.get_state_snapshot_dict()

        for state_id, handles in state_snapshot.items():
            if object_handle in handles:
                object_states[state_id] = handles[object_handle]

        state_infos: List[ObjectStateInfo] = []
        all_object_states = self.get_all_states()
        for state, value in object_states.items():
            state_spec = all_object_states[state]
            state_infos.append(
                ObjectStateInfo(
                    state_spec=state_spec,
                    value=value,
                )
            )

        return state_infos

    def get_category_from_handle(self, object_handle: str) -> Optional[str]:
        """
        Get the semantic category name of a given object handle.
        Returns None if the object category is unknown.
        """
        return self._object_handle_categories.get(object_handle, None)

    def handle_to_id(self, handle: str) -> Optional[int]:
        """
        Get the object_id associated to a given object handle.
        Note that for articulated objects, all children (links) have different IDs but the same handle.
        """
        return self._handle_to_id.get(handle, None)

    def id_to_handle(self, object_id: int) -> Optional[str]:
        """
        Get the object handle associated to a given object id.
        Note that for articulated objects, all children (links) have different IDs but the same handle.
        """
        return self._id_to_handle.get(object_id, None)

    def get_rigid_object(
        self, object_id: int
    ) -> Optional[ManagedBulletRigidObject]:
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
            and obj.is_articulated
            and object_id in obj.link_object_ids
        ):
            return obj.link_object_ids[object_id]
        return None

    def get_agent_object_ids(self, agent_index: int) -> Set[int]:
        """Get the IDs of objects composing an agent (including links)."""
        # TODO: Cache
        sim = self._sim
        agent_manager = sim.agents_mgr
        agent_object_ids: Set[int] = set()
        agent = agent_manager[agent_index]
        agent_ao = agent.articulated_agent.sim_obj
        agent_object_ids.add(agent_ao.object_id)

        for link_object_id in agent_ao.link_object_ids:
            agent_object_ids.add(link_object_id)

        return agent_object_ids

    def get_primary_object_region(
        self,
        obj: Union[ManagedBulletArticulatedObject, ManagedBulletRigidObject],
    ) -> Optional[SemanticRegion]:
        """Get the semantic region that contains most of an object."""
        object_regions = sim_utilities.get_object_regions(
            sim=self._sim, object_a=obj, ao_link_map=self._link_id_to_ao_map
        )
        if len(object_regions) > 0:
            primary_region = object_regions[0]
            region_name = primary_region[0]
            return self._regions[region_name]
        return None

    def is_any_agent_holding_object(self, object_id: int) -> bool:
        """
        Checks whether the specified object is being held by an agent.
        This function looks up both the HITL world state and grasp managers.
        """
        sim = self._sim
        agents_mgr = sim.agents_mgr

        for agent_index in range(len(agents_mgr.agent_names)):
            grasp_mgr = agents_mgr._all_agent_data[agent_index].grasp_mgr
            if grasp_mgr._snapped_obj_id == object_id:
                return True

        return object_id in self._all_held_object_ids

    def _try_initialize_object_states(
        self, sim: RearrangeSim, config: DictConfig
    ) -> None:
        """
        Try initializing object states and categories. Depends on the 'habitat_llm' external package.
        """
        try:
            # Initialize object state machine.
            from habitat_llm.sims.collaboration_sim import (
                initialize_object_state_machine,
            )
            from habitat_llm.sims.metadata_interface import (
                MetadataInterface,
                get_metadata_dict_from_config,
            )

            metadata_dict = get_metadata_dict_from_config(
                dataset_config=config.habitat.dataset
            )
            metadata_interface = MetadataInterface(metadata_dict)
            metadata_interface.refresh_scene_caches(
                sim, filter_receptacles=True
            )
            self._metadata_interface = metadata_interface
            osm = initialize_object_state_machine(sim, metadata_interface)
            self._object_state_machine = osm

            # Get object semantic categories.
            all_objects = sim_utilities.get_all_objects(sim)
            for obj in all_objects:
                self._object_handle_categories[
                    obj.handle
                ] = metadata_interface.get_object_instance_category(obj)

            # Get active object states.
            active_states = osm.active_states
            active_states_dict: Dict[str, ObjectStateSpec] = {}
            for active_state in active_states:
                active_states_dict[active_state.name] = active_state
            self._all_states = active_states_dict
        except Exception as e:
            print(f"Object states could not be loaded. {e}")
            self._object_state_machine = None

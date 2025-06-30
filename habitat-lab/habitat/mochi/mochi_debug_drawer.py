# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

raise RuntimeError("work in progress!")

import numpy as np
import numpy.typing as npt
import dataclasses

from mochi_gym.utils.mochi_platform import mochi_agents_api

class MochiDebugDrawer:
    def __init__(self, mochi):

        self._show_contacts = True
        self._mochi = mochi

        self._contact_points = None
        self._contact_forces = None

        
        pass

    @dataclasses.dataclass
    class ActorMesh:
        actor: mochi_agents_api.ActorHandle
        coordinates: npt.NDArray[float]
        connectivity: npt.NDArray[int]
        min_coord: npt.NDArray[float]
        max_coord: npt.NDArray[float]

    # copied from mochi_gym/mochi_gym/renderers/renderer.py
    def _initialize_render_meshes(self):
        # Initialize render meshes.
        self._meshes = []

        actors = self._mochi.get_actors_names()

        for actor_name in actors:
            actor_handle = self._mochi.get_actor_handle(actor_name)
            coordinates = self._mochi.get_actor_surface_mesh_coordinates(actor_handle)
            coordinates = np.array(coordinates, dtype=np.float32).reshape(-1, 3)
            connectivity = self._mochi.get_actor_surface_mesh_connectivity(actor_handle)
            connectivity = np.array(connectivity, dtype=np.int32).reshape(-1, 3)

            actor_mesh = MochiDebugDrawer.ActorMesh(
                actor=actor_handle,
                coordinates=coordinates,
                connectivity=connectivity,
                min_coord=np.min(coordinates, axis=0),
                max_coord=np.max(coordinates, axis=0),
            )
            self._meshes.append(actor_mesh)

        # Compute rough scale of the scene from the bounding box of the render meshes.
        aggregate_min_coords = np.vstack([m.min_coord for m in self._meshes])
        aggregate_max_coords = np.vstack([m.max_coord for m in self._meshes])
        min_coord = np.min(aggregate_min_coords, axis=0)
        max_coord = np.max(aggregate_max_coords, axis=0)
        self._scale = np.max(max_coord - min_coord)

    def update_post_step(self):

        if not self._show_contacts:
            return

        contact_points = []
        contact_forces = []
        for actor_mesh in self._meshes:
            actor_handle = actor_mesh.actor
            if not self._mochi.is_actor_contact_query_enabled(actor_handle):
                continue
            if not self._mochi.get_num_contacts(actor_handle):
                continue

            actor_contact_indices, actor_contact_forces = (
                self._mochi.get_contact_forces(actor_handle)
            )
            actor_contact_points = actor_mesh.coordinates[actor_contact_indices, :]
            contact_points.append(actor_contact_points)
            contact_forces.append(actor_contact_forces)

        if len(contact_points) > 0:
            contact_points = np.vstack(contact_points)
            contact_forces = np.vstack(contact_forces)        

        self._contact_points = contact_points
        self._contact_forces = contact_forces

    def _draw_meshes

    def debug_draw(self, line_render):

        pass
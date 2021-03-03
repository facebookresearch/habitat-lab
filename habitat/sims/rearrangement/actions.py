#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import attr
import magnum as mn

import habitat_sim
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat_sim.agent.controls.controls import ActuationSpec


def raycast(sim, sensor_name, crosshair_pos=[128, 128], max_distance=2.0):
    r"""Cast a ray in the direction of crosshair and check if it collides
    with another object within a certain distance threshold
    :param sim: Simulator object
    :param sensor_name: name of the visual sensor to be used for raycasting
    :param crosshair_pos: 2D coordiante in the viewport towards which the
        ray will be cast
    :param max_distance: distance threshold beyond which objects won't
        be considered
    """
    visual_sensor = sim._sensors[sensor_name]
    scene_graph = sim.get_active_scene_graph()
    scene_graph.set_default_render_camera_parameters(
        visual_sensor._sensor_object
    )
    render_camera = scene_graph.get_default_render_camera()
    center_ray = render_camera.unproject(mn.Vector2i(crosshair_pos))

    raycast_results = sim.cast_ray(center_ray, max_distance=max_distance)

    closest_object = -1
    closest_dist = 1000.0
    if raycast_results.has_hits():
        for hit in raycast_results.hits:
            if hit.ray_distance < closest_dist:
                closest_dist = hit.ray_distance
                closest_object = hit.object_id

    return closest_object


@attr.s(auto_attribs=True, slots=True)
class GrabReleaseActuationSpec(ActuationSpec):
    visual_sensor_name: str = "rgb"
    crosshair_pos: List[int] = [128, 128]
    amount: float = 2.0


@registry.register_action_space_configuration(name="RearrangementActions-v0")
class RearrangementSimV0ActionSpaceConfiguration(
    HabitatSimV1ActionSpaceConfiguration
):
    def __init__(self, config):
        super().__init__(config)
        if not HabitatSimActions.has_action("GRAB_RELEASE"):
            HabitatSimActions.extend_action_space("GRAB_RELEASE")

    def get(self):
        config = super().get()
        new_config = {
            HabitatSimActions.GRAB_RELEASE: habitat_sim.ActionSpec(
                "grab_or_release_object_under_crosshair",
                GrabReleaseActuationSpec(
                    visual_sensor_name=self.config.VISUAL_SENSOR,
                    crosshair_pos=self.config.CROSSHAIR_POS,
                    amount=self.config.GRAB_DISTANCE,
                ),
            )
        }
        config.update(new_config)

        return config

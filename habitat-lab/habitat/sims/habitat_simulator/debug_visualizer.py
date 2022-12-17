#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Union

import magnum as mn
import numpy as np

import habitat_sim
from habitat.core.logging import logger
from habitat.utils.common import check_make_dir


class DebugVisualizer:
    """
    Support class for simple visual debugging of a Simulator instance.
    Assumes the default agent (0) is a camera (i.e. there exists an RGB sensor coincident with agent 0 transformation).
    """

    def __init__(
        self,
        sim: habitat_sim.Simulator,
        output_path: str = "visual_debug_output/",
        default_sensor_uuid: str = "rgb",
    ) -> None:
        """
        Initialize the debugger provided a Simulator and the uuid of the debug sensor.
        NOTE: Expects the debug sensor attached to and coincident with agent 0's frame.
        """
        self.sim = sim
        self.output_path = output_path
        self.default_sensor_uuid = default_sensor_uuid
        self._debug_obs: List[Any] = []
        # NOTE: visualizations from the DebugLinerRender utility will only be visible in PINHOLE RGB sensor views
        self.debug_line_render = sim.get_debug_line_render()

    def look_at(
        self,
        look_at: mn.Vector3,
        look_from: Optional[mn.Vector3] = None,
        look_up: Optional[mn.Vector3] = None,
    ) -> None:
        """
        Point the debug camera at a target.
        """
        agent = self.sim.get_agent(0)
        camera_pos = (
            look_from
            if look_from is not None
            else agent.scene_node.translation
        )
        if look_up is None:
            # pick a valid "up" vector.
            look_dir = look_at - camera_pos
            look_up = (
                mn.Vector3(0, 1.0, 0)
                if look_dir[0] != 0 or look_dir[2] != 0
                else mn.Vector3(1.0, 0, 0)
            )
        agent.scene_node.rotation = mn.Quaternion.from_matrix(
            mn.Matrix4.look_at(camera_pos, look_at, look_up).rotation()
        )
        agent.scene_node.translation = camera_pos

    def get_observation(
        self,
        look_at: Optional[mn.Vector3] = None,
        look_from: Optional[mn.Vector3] = None,
        obs_cache: Optional[List[Any]] = None,
    ) -> None:
        """
        Render a debug observation of the current state and cache it.
        Optionally configure the camera transform.
        Optionally provide an alternative observation cache.
        """
        if look_at is not None:
            self.look_at(look_at, look_from)
        if obs_cache is None:
            self._debug_obs.append(self.sim.get_sensor_observations())
        else:
            obs_cache.append(self.sim.get_sensor_observations())

    def save_observation(
        self,
        output_path: Optional[str] = None,
        prefix: str = "",
        look_at: Optional[mn.Vector3] = None,
        look_from: Optional[mn.Vector3] = None,
        obs_cache: Optional[List[Any]] = None,
        show: bool = True,
    ) -> str:
        """
        Get an observation and save it to file.
        Return the filepath.
        """
        obs_cache = []
        self.get_observation(look_at, look_from, obs_cache)
        # save the obs as an image
        if output_path is None:
            output_path = self.output_path
        check_make_dir(output_path)
        from habitat_sim.utils import viz_utils as vut

        image = vut.observation_to_image(obs_cache[0]["rgb"], "color")
        from datetime import datetime

        # filename format "prefixmonth_day_year_hourminutesecondmicrosecond.png"
        date_time = datetime.now().strftime("%m_%d_%Y_%H%M%S%f")
        file_path = output_path + prefix + date_time + ".png"
        image.save(file_path)
        if show:
            image.show()
        return file_path

    def peek_rigid_object(
        self,
        obj: habitat_sim.physics.ManagedRigidObject,
        cam_local_pos: Optional[mn.Vector3] = None,
        peek_all_axis: bool = False,
    ) -> str:
        """
        Specialization to peek a rigid object.
        See _peek_object.
        """

        return self._peek_object(
            obj,
            obj.root_scene_node.cumulative_bb,
            cam_local_pos,
            peek_all_axis,
        )

    def peek_articulated_object(
        self,
        obj: habitat_sim.physics.ManagedArticulatedObject,
        cam_local_pos: Optional[mn.Vector3] = None,
        peek_all_axis: bool = False,
    ) -> str:
        """
        Specialization to peek an articulated object.
        See _peek_object.
        """
        from habitat.sims.habitat_simulator.sim_utilities import (
            get_ao_global_bb,
        )

        obj_bb = get_ao_global_bb(obj)

        return self._peek_object(obj, obj_bb, cam_local_pos, peek_all_axis)

    def _peek_object(
        self,
        obj: Union[
            habitat_sim.physics.ManagedArticulatedObject,
            habitat_sim.physics.ManagedRigidObject,
        ],
        obj_bb: mn.Range3D,
        cam_local_pos: Optional[mn.Vector3] = None,
        peek_all_axis: bool = False,
    ) -> str:
        """
        Compute a camera placement to view an ArticulatedObject and show/save an observation.
        Return the filepath.
        If peek_all_axis, then create a merged 3x2 matrix of images looking at the object from all angles.
        """
        obj_abs_transform = obj.root_scene_node.absolute_transformation()
        look_at = obj_abs_transform.translation
        bb_size = obj_bb.size()
        # TODO: query fov and aspect from the camera spec
        fov = 90
        aspect = 0.75
        import math

        # compute the optimal view distance from the camera specs and object size
        distance = (np.amax(np.array(bb_size)) / aspect) / math.tan(
            fov / (360 / math.pi)
        )
        if cam_local_pos is None:
            # default to -Z (forward) of the object
            cam_local_pos = mn.Vector3(0, 0, -1)
        if not peek_all_axis:
            look_from = (
                obj_abs_transform.transform_vector(cam_local_pos).normalized()
                * distance
                + look_at
            )
            return self.save_observation(
                prefix="peek_" + obj.handle,
                look_at=look_at,
                look_from=look_from,
            )
        else:
            # collect axis observations
            axis_obs: List[Any] = []
            for axis in range(6):
                axis_vec = mn.Vector3()
                axis_vec[axis % 3] = 1 if axis // 3 == 0 else -1
                look_from = (
                    obj_abs_transform.transform_vector(axis_vec).normalized()
                    * distance
                    + look_at
                )
                self.get_observation(look_at, look_from, axis_obs)
            # stitch images together
            stitched_image = None
            from PIL import Image

            from habitat_sim.utils import viz_utils as vut

            for ix, obs in enumerate(axis_obs):
                image = vut.observation_to_image(obs["rgb"], "color")
                if stitched_image is None:
                    stitched_image = Image.new(
                        image.mode, (image.size[0] * 3, image.size[1] * 2)
                    )
                location = (
                    image.size[0] * (ix % 3),
                    image.size[1] * (0 if ix // 3 == 0 else 1),
                )
                stitched_image.paste(image, location)
            stitched_image.show()
        return ""

    def make_debug_video(
        self,
        output_path: Optional[str] = None,
        prefix: str = "",
        fps: int = 4,
        obs_cache: Optional[List[Any]] = None,
    ) -> None:
        """
        Produce a video from a set of debug observations.
        """
        if output_path is None:
            output_path = self.output_path

        check_make_dir(output_path)

        # get a timestamp tag with current date and time for video name
        from datetime import datetime

        date_time = datetime.now().strftime("%m_%d_%Y_%H%M%S")

        if obs_cache is None:
            obs_cache = self._debug_obs

        from habitat_sim.utils import viz_utils as vut

        file_path = output_path + prefix + date_time
        logger.info(f"DebugVisualizer: Saving debug video to {file_path}")
        vut.make_video(
            obs_cache, self.default_sensor_uuid, "color", file_path, fps=fps
        )

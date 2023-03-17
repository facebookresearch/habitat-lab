#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, List, Optional, Tuple, Union

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

        :param sim: Simulator instance must be provided for attachment.
        :param output_path: Directory path for saving debug images and videos.
        :param default_sensor_uuid: Which sensor uuid to use for debug image rendering.
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
        Standard look_at function syntax.

        :param look_at: 3D global position to point the camera towards.
        :param look_from: 3D global position of the camera.
        :param look_up: 3D global "up" vector for aligning the camera roll.
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

        :param look_at: 3D global position to point the camera towards.
        :param look_from: 3D global position of the camera.
        :param obs_cache: Optioanlly provide an external observation cache datastructure in place of self._debug_obs.
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
        Render an observation and save it to file.
        Return the filepath.

        :param output_path: Optional directory path for saving debug images and videos. Otherwise use self.output_path.
        :param prefix: Optional prefix for output filename. Filename format: "<prefix>month_day_year_hourminutesecondmicrosecond.png"
        :param look_at: 3D global position to point the camera towards.
        :param look_from: 3D global position of the camera.
        :param obs_cache: Optioanlly provide an external observation cache datastructure in place of self._debug_obs.
        :param show: If True, open the image immediately.
        """
        obs_cache = []
        self.get_observation(look_at, look_from, obs_cache)
        # save the obs as an image
        if output_path is None:
            output_path = self.output_path
        check_make_dir(output_path)
        from habitat_sim.utils import viz_utils as vut

        image = vut.observation_to_image(
            obs_cache[0][self.default_sensor_uuid], "color"
        )
        from datetime import datetime

        # filename format "prefixmonth_day_year_hourminutesecondmicrosecond.png"
        date_time = datetime.now().strftime("%m_%d_%Y_%H%M%S%f")
        file_path = os.path.join(output_path, prefix + date_time + ".png")
        image.save(file_path)
        if show:
            image.show()
        return file_path

    def render_debug_lines(
        self,
        debug_lines: Optional[List[Tuple[List[mn.Vector3], mn.Color4]]] = None,
    ) -> None:
        """
        Draw a set of debug lines with accompanying colors.

        :param debug_lines: A set of debug line strips with accompanying colors. Each list entry contains a list of points and a color.
        """
        # support None input to make useage easier elsewhere
        if debug_lines is not None:
            for points, color in debug_lines:
                for p_ix, point in enumerate(points):
                    if p_ix == 0:
                        continue
                    prev_point = points[p_ix - 1]
                    self.debug_line_render.draw_transformed_line(
                        prev_point,
                        point,
                        color,
                    )

    def render_debug_circles(
        self,
        debug_circles: Optional[
            List[Tuple[mn.Vector3, float, mn.Vector3, mn.Color4]]
        ] = None,
    ) -> None:
        """
        Draw a set of debug circles with accompanying colors.

        :param debug_circles: A list of debug line render circle Tuples, each with (center, radius, normal, color).
        """
        # support None input to make useage easier elsewhere
        if debug_circles is not None:
            for center, radius, normal, color in debug_circles:
                self.debug_line_render.draw_circle(
                    translation=center,
                    radius=radius,
                    color=color,
                    num_segments=12,
                    normal=normal,
                )

    def peek_rigid_object(
        self,
        obj: habitat_sim.physics.ManagedRigidObject,
        cam_local_pos: Optional[mn.Vector3] = None,
        peek_all_axis: bool = False,
        additional_savefile_prefix="",
        debug_lines: Optional[List[Tuple[List[mn.Vector3], mn.Color4]]] = None,
        debug_circles: Optional[
            List[Tuple[mn.Vector3, float, mn.Vector3, mn.Color4]]
        ] = None,
        show: bool = False,
    ) -> str:
        """
        Helper function to generate image(s) of an object for contextual debugging purposes.
        Specialization to peek a rigid object. See _peek_object.
        Compute a camera placement to view an object. Show/save an observation. Return the filepath.

        :param obj: The ManagedRigidObject to peek.
        :param cam_local_pos: Optionally provide a camera location in location local coordinates. Otherwise offset along local -Z axis from the object.
        :param peek_all_axis: Optionally create a merged 3x2 matrix of images looking at the object from all angles.
        :param additional_savefile_prefix: Optionally provide an additional prefix for the save filename to differentiate the images.
        :param debug_lines: Optionally provide a list of debug line render tuples, each with a list of points and a color. These will be displayed in all peek images.
        :param debug_circles: Optionally provide a list of debug line render circle Tuples, each with (center, radius, normal, color). These will be displayed in all peek images.
        :param show: If True, open and display the image immediately.
        """

        return self._peek_object(
            obj,
            obj.root_scene_node.cumulative_bb,
            cam_local_pos,
            peek_all_axis,
            additional_savefile_prefix,
            debug_lines,
            debug_circles,
            show,
        )

    def peek_articulated_object(
        self,
        obj: habitat_sim.physics.ManagedArticulatedObject,
        cam_local_pos: Optional[mn.Vector3] = None,
        peek_all_axis: bool = False,
        additional_savefile_prefix="",
        debug_lines: Optional[List[Tuple[List[mn.Vector3], mn.Color4]]] = None,
        debug_circles: Optional[
            List[Tuple[mn.Vector3, float, mn.Vector3, mn.Color4]]
        ] = None,
        show: bool = False,
    ) -> str:
        """
        Helper function to generate image(s) of an object for contextual debugging purposes.
        Specialization to peek an articulated object. See _peek_object.
        Compute a camera placement to view an object. Show/save an observation. Return the filepath.

        :param obj: The ManagedArticulatedObject to peek.
        :param cam_local_pos: Optionally provide a camera location in location local coordinates. Otherwise offset along local -Z axis from the object.
        :param peek_all_axis: Optionally create a merged 3x2 matrix of images looking at the object from all angles.
        :param additional_savefile_prefix: Optionally provide an additional prefix for the save filename to differentiate the images.
        :param debug_lines: Optionally provide a list of debug line render tuples, each with a list of points and a color. These will be displayed in all peek images.
        :param debug_circles: Optionally provide a list of debug line render circle Tuples, each with (center, radius, normal, color). These will be displayed in all peek images.
        :param show: If True, open and display the image immediately.
        """
        from habitat.sims.habitat_simulator.sim_utilities import (
            get_ao_global_bb,
        )

        obj_bb = get_ao_global_bb(obj)

        return self._peek_object(
            obj,
            obj_bb,
            cam_local_pos,
            peek_all_axis,
            additional_savefile_prefix,
            debug_lines,
            debug_circles,
            show,
        )

    def _peek_object(
        self,
        obj: Union[
            habitat_sim.physics.ManagedArticulatedObject,
            habitat_sim.physics.ManagedRigidObject,
        ],
        obj_bb: mn.Range3D,
        cam_local_pos: Optional[mn.Vector3] = None,
        peek_all_axis: bool = False,
        additional_savefile_prefix="",
        debug_lines: Optional[List[Tuple[List[mn.Vector3], mn.Color4]]] = None,
        debug_circles: Optional[
            List[Tuple[mn.Vector3, float, mn.Vector3, mn.Color4]]
        ] = None,
        show: bool = False,
    ) -> str:
        """
        Internal helper function to generate image(s) of an object for contextual debugging purposes.
        Compute a camera placement to view an object. Show/save an observation. Return the filepath.

        :param obj: The ManagedRigidObject or ManagedArticulatedObject to peek.
        :param obj_bb: The object's bounding box (provided by consumer functions.)
        :param cam_local_pos: Optionally provide a camera location in location local coordinates. Otherwise offset along local -Z axis from the object.
        :param peek_all_axis: Optionally create a merged 3x2 matrix of images looking at the object from all angles.
        :param additional_savefile_prefix: Optionally provide an additional prefix for the save filename to differentiate the images.
        :param debug_lines: Optionally provide a list of debug line render tuples, each with a list of points and a color. These will be displayed in all peek images.
        :param debug_circles: Optionally provide a list of debug line render circle Tuples, each with (center, radius, normal, color). These will be displayed in all peek images.
        :param show: If True, open and display the image immediately.
        """
        obj_abs_transform = obj.root_scene_node.absolute_transformation()
        return self._peek_bb(
            bb_name=obj.handle,
            bb=obj_bb,
            world_transform=obj_abs_transform,
            cam_local_pos=cam_local_pos,
            peek_all_axis=peek_all_axis,
            additional_savefile_prefix=additional_savefile_prefix,
            debug_lines=debug_lines,
            debug_circles=debug_circles,
            show=show,
        )

    def peek_scene(
        self,
        cam_local_pos: Optional[mn.Vector3] = None,
        peek_all_axis: bool = False,
        additional_savefile_prefix="",
        debug_lines: Optional[List[Tuple[List[mn.Vector3], mn.Color4]]] = None,
        debug_circles: Optional[
            List[Tuple[mn.Vector3, float, mn.Vector3, mn.Color4]]
        ] = None,
        show: bool = False,
    ) -> str:
        """
        Helper function to generate image(s) of the scene for contextual debugging purposes.
        Specialization to peek a scene. See _peek_bb.
        Compute a camera placement to view the scene. Show/save an observation. Return the filepath.

        :param cam_local_pos: Optionally provide a camera location in location local coordinates. Otherwise offset along local -Z axis from the object.
        :param peek_all_axis: Optionally create a merged 3x2 matrix of images looking at the object from all angles.
        :param additional_savefile_prefix: Optionally provide an additional prefix for the save filename to differentiate the images.
        :param debug_lines: Optionally provide a list of debug line render tuples, each with a list of points and a color. These will be displayed in all peek images.
        :param debug_circles: Optionally provide a list of debug line render circle Tuples, each with (center, radius, normal, color). These will be displayed in all peek images.
        :param show: If True, open and display the image immediately.
        """
        return self._peek_bb(
            bb_name=self.sim.curr_scene_name,
            bb=self.sim.get_active_scene_graph().get_root_node().cumulative_bb,
            world_transform=mn.Matrix4.identity_init(),
            cam_local_pos=cam_local_pos,
            peek_all_axis=peek_all_axis,
            additional_savefile_prefix=additional_savefile_prefix,
            debug_lines=debug_lines,
            debug_circles=debug_circles,
            show=show,
        )

    def _peek_bb(
        self,
        bb_name: str,
        bb: mn.Range3D,
        world_transform: Optional[mn.Matrix4] = None,
        cam_local_pos: Optional[mn.Vector3] = None,
        peek_all_axis: bool = False,
        additional_savefile_prefix="",
        debug_lines: Optional[List[Tuple[List[mn.Vector3], mn.Color4]]] = None,
        debug_circles: Optional[
            List[Tuple[mn.Vector3, float, mn.Vector3, mn.Color4]]
        ] = None,
        show: bool = False,
    ) -> str:
        """
        Internal helper function to generate image(s) of any bb for contextual debugging purposes.
        Compute a camera placement to view the bb. Show/save an observation. Return the filepath.

        :param bb_name: The name of the entity we're peeking for filepath naming.
        :param bb: The entity's bounding box (provided by consumer functions.)
        :param world_transform: The entity's world transform provided by consumer functions, default identity.
        :param cam_local_pos: Optionally provide a camera location in location local coordinates. Otherwise offset along local -Z axis from the object.
        :param peek_all_axis: Optionally create a merged 3x2 matrix of images looking at the object from all angles.
        :param additional_savefile_prefix: Optionally provide an additional prefix for the save filename to differentiate the images.
        :param debug_lines: Optionally provide a list of debug line render tuples, each with a list of points and a color. These will be displayed in all peek images.
        :param debug_circles: Optionally provide a list of debug line render circle Tuples, each with (center, radius, normal, color). These will be displayed in all peek images.
        :param show: If True, open and display the image immediately.
        """
        if world_transform is None:
            world_transform = mn.Matrix4.identity_init()
        look_at = world_transform.transform_point(bb.center())
        bb_size = bb.size()
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
                world_transform.transform_vector(cam_local_pos).normalized()
                * distance
                + look_at
            )
            self.render_debug_lines(debug_lines)
            self.render_debug_circles(debug_circles)
            return self.save_observation(
                prefix=additional_savefile_prefix + "peek_" + bb_name,
                look_at=look_at,
                look_from=look_from,
                show=show,
            )

        # collect axis observations
        axis_obs: List[Any] = []
        for axis in range(6):
            axis_vec = mn.Vector3()
            axis_vec[axis % 3] = 1 if axis // 3 == 0 else -1
            look_from = (
                world_transform.transform_vector(axis_vec).normalized()
                * distance
                + look_at
            )
            self.render_debug_lines(debug_lines)
            self.render_debug_circles(debug_circles)
            self.get_observation(look_at, look_from, axis_obs)
        # stitch images together
        stitched_image = None
        from PIL import Image

        from habitat_sim.utils import viz_utils as vut

        for ix, obs in enumerate(axis_obs):
            image = vut.observation_to_image(
                obs[self.default_sensor_uuid], "color"
            )
            if stitched_image is None:
                stitched_image = Image.new(
                    image.mode, (image.size[0] * 3, image.size[1] * 2)
                )
            location = (
                image.size[0] * (ix % 3),
                image.size[1] * (0 if ix // 3 == 0 else 1),
            )
            stitched_image.paste(image, location)
        if show:
            stitched_image.show()
        save_path = os.path.join(
            self.output_path,
            additional_savefile_prefix + "peek_6x_" + bb_name + ".png",
        )
        stitched_image.save(save_path)
        return save_path

    def make_debug_video(
        self,
        output_path: Optional[str] = None,
        prefix: str = "",
        fps: int = 4,
        obs_cache: Optional[List[Any]] = None,
    ) -> None:
        """
        Produce and save a video from a set of debug observations.

        :param output_path: Optional directory path for saving the video. Otherwise use self.output_path.
        :param prefix: Optional prefix for output filename. Filename format: "<output_path><prefix><timestamp>"
        :param fps: Framerate of the video. Defaults to 4FPS expecting disjoint still frames.
        :param obs_cache: Optioanlly provide an external observation cache datastructure in place of self._debug_obs.
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

        file_path = os.path.join(output_path, prefix + date_time)
        logger.info(f"DebugVisualizer: Saving debug video to {file_path}")
        vut.make_video(
            obs_cache, self.default_sensor_uuid, "color", file_path, fps=fps
        )

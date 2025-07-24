#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import magnum as mn
import numpy as np

from habitat.isaac_sim import isaac_prim_utils
from scripts.xr_pose_adapter import XRPose

if TYPE_CHECKING:
    from examples.hitl.isaac_robot_teleop.isaac_robot_teleop import (
        AppStateIsaacSimViewer,
    )


# events triggered from the UI which should be cached in metadata
class FrameEvent(Enum):
    # this event indicates the objects were reset to their initial positions from the episode
    RESET_OBJECTS = 1
    # this event indicates the robot's arm or finger joint motor positions were updated manually (not via IK from XR)
    RESET_ARMS_FINGERS = 2
    # this event indicates the robot's base position has been updated manually
    TELEPORT = 3
    # this event indicates a discontinuity in the XR state due to user re-syncing their current state to the robot cursor origin
    SYNC_XR_OFFSET = 4

    def __str__(self):
        return self.name.lower()


class FrameRecorder:
    """
    Data recording class encapsulating per-frame state records for use with the SessionRecorder API.
    """

    def __init__(self, app_state: "AppStateIsaacSimViewer"):
        self._app_state = app_state

        # sequentially cached frame data
        self.frame_data: List[Dict[str, Any]] = []

        # mutually exclusive execution modes
        self._recording = False
        self._replaying = False
        # NOTE: this keeps continuous time independent of discrete frame but is overwritten with frame is explicitly set
        self._replay_time: float = 0
        self._replay_frame: int = 0
        # if true, replay will loop when complete
        self._looping = True
        # optional variable to set a custom start frame for looping
        self._start_frame: int = None

    @property
    def recording(self):
        return self._recording

    @recording.setter
    def recording(self, do_recording: bool):
        self._recording = do_recording
        if self._recording:
            self._replaying = False

    @property
    def replaying(self):
        return self._replaying

    @replaying.setter
    def replaying(self, do_replay: bool):
        self._replaying = do_replay
        if self._replaying:
            self._recording = False

    @property
    def replay_frame(self):
        return self._replay_frame

    @replay_frame.setter
    def replay_frame(self, replay_frame: int):
        """
        Sets and validates the replay frame, but does not change the state.
        """
        if replay_frame >= len(self.frame_data):
            raise IndexError(
                f"Requested replay frame '{replay_frame}' out of range for loaded {len(self.frame_data)} frames."
            )
        self._replay_frame = replay_frame
        self._replay_time = self.frame_data[self._replay_frame]["t"]

    @property
    def replay_time(self) -> float:
        if len(self.frame_data) == 0:
            return 0.0
        return self._replay_time

    @replay_time.setter
    def replay_time(self, replay_time: float):
        """
        Sets replay frame to the closest frame to that time.
        """
        num_frames = len(self.frame_data)
        if num_frames == 0:
            return

        # binary search for frame with time t
        l = 0
        r = len(self.frame_data) - 1
        m = (r + l) // 2
        while r > l:
            if self.frame_data[m]["t"] > replay_time:
                r = m - 1
            elif self.frame_data[m]["t"] < replay_time:
                l = m + 1
            else:
                break
            m = (r + l) // 2
        self._replay_frame = m
        self._replay_time = replay_time

    def get_robot_state(self):
        """
        Scrape the state from the Robot object.
        """

        robot = self._app_state.robot
        pos, rot = robot.get_root_pose()
        base_linear_velocity = isaac_prim_utils.usd_to_habitat_position(
            robot._robot.get_linear_velocity()
        )
        base_linear_velocity = [float(val) for val in base_linear_velocity]
        base_angular_velocity = isaac_prim_utils.usd_to_habitat_position(
            robot._robot.get_angular_velocity()
        )
        base_angular_velocity = [float(val) for val in base_angular_velocity]
        robot_state = {
            # base state
            "base_pos": np.array(pos).tolist(),
            "base_rotation_quat": isaac_prim_utils.magnum_quat_to_list_wxyz(
                rot
            ),
            "base_rotation_angle": robot.base_rot,
            "base_linear_velocity": base_linear_velocity,
            "base_angular_velocity": base_angular_velocity,
            # joint states
            "joint_positions": robot._robot.get_joint_positions().tolist(),
            "joint_velocities": robot._robot.get_joint_velocities().tolist(),
            # motor states
            "motor_targets": robot._robot.get_applied_action().get_dict()[
                "joint_positions"
            ],
        }

        return robot_state

    def get_object_states(self):
        """
        Scrape the dynamic state from all added rigid objects.
        #TODO: also scrape AO states from the furniture.
        """
        object_states = []
        active_rigid_objects = self._app_state._rigid_objects
        for ro in active_rigid_objects:
            object_states.append(
                {
                    "prim_path": ro._rigid_prim.prim_path,
                    "object_id": ro.object_id,
                    "transformation": np.array(ro.transformation).tolist(),
                    "angular_velocity": np.array(ro.angular_velocity).tolist(),
                    "linear_velocity": np.array(ro.linear_velocity).tolist(),
                }
            )
        return object_states

    def get_xr_state(self):
        """
        Get the json rep of the global XRPose of connected xr user or an empty dict if no user is connected.
        This pose includes the position and orientation of both hands (controllers) and the headset.
        """
        current_xr_pose = XRPose(
            remote_client_state=self._app_state._app_service.remote_client_state
        )
        if current_xr_pose.valid:
            return self._app_state.xr_pose_adapter.get_global_xr_pose(
                current_xr_pose
            ).to_json_dict()
        return {}

    def record_state(
        self,
        elapsed_time: float,
        frame_events: Optional[List[FrameEvent]] = None,
    ) -> None:
        """
        Scrapes the app for relevant per-frame state information and aggregates it in a dict.
        """

        _frame_data: Dict[str, Any] = {
            "t": elapsed_time,
            "users": [],  # TODO
            "object_states": self.get_object_states(),
            "robot_state": self.get_robot_state(),
            "xr_state": self.get_xr_state(),
            "events": frame_events if frame_events is not None else [],
        }
        # TODO: user camera and cursor states
        self.frame_data.append(_frame_data)

    def set_object_states(self, obj_states: List[Dict[str, Any]]):
        """
        Applies the object states cached in a recorded frame data dict.
        See get_object_states
        #TODO: AO states
        """
        for object_state_record in obj_states:
            ro = self._app_state._isaac_rom.get_object_by_id(
                object_state_record["object_id"]
            )
            assert ro._rigid_prim.prim_path == object_state_record["prim_path"]
            ro_t = mn.Matrix4(
                [
                    [
                        object_state_record["transformation"][j][i]
                        for j in range(4)
                    ]
                    for i in range(4)
                ]
            )
            try:
                ro.transformation = ro_t
            except ValueError as e:
                print(
                    f"Failed to set object transform '{ro._rigid_prim.prim_path}' with error: {e}"
                )
            ro.angular_velocity = mn.Vector3(
                *object_state_record["angular_velocity"]
            )
            ro.linear_velocity = mn.Vector3(
                *object_state_record["linear_velocity"]
            )

    def set_robot_state(self, robot_state_dict: Dict[str, Any]):
        """
        Applies the robot state cached in a recorded frame data dict.
        See get_robot_state
        """
        robot = self._app_state.robot
        robot.set_root_pose(mn.Vector3(*robot_state_dict["base_pos"]))
        robot.base_rot = robot_state_dict["base_rotation_angle"]
        robot._robot.set_linear_velocity(
            isaac_prim_utils.habitat_to_usd_position(
                robot_state_dict["base_linear_velocity"]
            )
        )
        robot._robot.set_angular_velocity(
            isaac_prim_utils.habitat_to_usd_position(
                robot_state_dict["base_angular_velocity"]
            )
        )
        robot._robot.set_joint_positions(
            np.array(robot_state_dict["joint_positions"])
        )
        robot._robot.set_joint_velocities(
            np.array(robot_state_dict["joint_velocities"])
        )
        from omni.isaac.core.utils.types import ArticulationAction

        action = ArticulationAction(
            joint_positions=np.array(robot_state_dict["motor_targets"])
        )
        robot._robot.apply_action(action)

    def apply_state(self, frame: int):
        """
        Applies the recorded frame state for self.replay_frame
        """
        frame_data = self.frame_data[frame]
        self.set_object_states(frame_data["object_states"])
        self.set_robot_state(frame_data["robot_state"])

    def get_xr_pose_from_frame(self, frame: int = None):
        if frame is None:
            frame = self.replay_frame

        return XRPose(json_pose_dict=self.frame_data[frame]["xr_state"])

    def update(
        self, t: float, frame_events: Optional[List[FrameEvent]] = None
    ) -> None:
        """
        Control loop callback.
        Either records a frame or replays a frame depending on the configured mode variables.
        """

        self.replay_time = t
        if self.recording:
            self.record_state(t, frame_events)
        elif self.replaying:
            # in this case t is the desired wall clock time to replay
            # NOTE: will match to one of the closest frames not an exact time
            if self._looping and self.replay_time > self.frame_data[-1]["t"]:
                self.replay_frame = (
                    0 if self._start_frame is None else self._start_frame
                )
            elif self._looping and (
                self.replay_time < self.frame_data[0]["t"]
                or self._start_frame is not None
                and self.replay_time < self.frame_data[self._start_frame]["t"]
            ):
                self.replay_frame = len(self.frame_data) - 1
            self.apply_state(self.replay_frame)

    def respace_frames(self):
        """
        Overwrites the frame times to evenly interpolate 1/30 sec between each frame.
        NOTE: introduced to correct for a bug which locked the frame time in pilot data.
        """
        print("!!!RE-SPACING FRAMES!!!")
        for ix, frame_data in enumerate(self.frame_data):
            frame_data["t"] = ix * (1.0 / 30)

    def save_json(self, filename: str = "frame_data.json"):
        """
        Serialize the frame data to JSON
        """

        with open(filename, "w") as f:
            json.dump(
                self.frame_data,
                f,
                indent=4,
            )
        print(
            f"Saved frame trajectory with {len(self.frame_data)} frames to {filename}"
        )

    def load_json(self, filename: str = "frame_data.json"):
        """
        Load a serialized frame data json for replay.
        """

        with open(filename, "r") as f:
            self.frame_data = json.load(f)
            # TODO: not necessary in long term, but needed for pilot
            self.respace_frames()

    def load_episode_record_json_gz(self, filepath: str) -> None:
        """
        Loads the serialized frame data from within an episode record (i.e. a crowd sourced session's data file)
        NOTE: returns the task prompt
        """

        with gzip.open(filepath, "rt") as f:
            replay_data = json.load(f)
            self.frame_data = replay_data["frames"]
            # TODO: not necessary in long term, but needed for pilot
            self.respace_frames()
            task_prompt = replay_data["episode"]["episode_info"]["task_prompt"]
            print(f"Task Prompt: {task_prompt}")
            return task_prompt

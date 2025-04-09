import copy
import json
import os
from typing import Any, Dict, List

import magnum as mn

from habitat_hitl.app_states.app_service import RemoteClientState
from habitat_sim.gfx import DebugLineRender
from scripts.robot import debug_draw_axis


class XRPose:
    """
    Encapsulates the XR controller and head pose states.
    """

    def __init__(
        self,
        remote_client_state: RemoteClientState = None,
        json_pose_dict: Dict[str, Any] = None,
    ):
        """
        Extracts the XR pose information from a RemoteClientState object or json dictionary
        If no XR input is available, self.valid is False.
        """

        self.pos_head: mn.Vector3 = None
        self.rot_head: mn.Quaternion = None
        self.pos_left: mn.Vector3 = None
        self.rot_left: mn.Quaternion = None
        self.pos_right: mn.Vector3 = None
        self.rot_right: mn.Quaternion = None

        if remote_client_state is not None and json_pose_dict is not None:
            self.valid = False
            print(
                "Error: cannot instantiate a pose from multiple sources, choose one."
            )
        if remote_client_state is None and json_pose_dict is None:
            self.valid = False

        if remote_client_state is not None:
            self.from_remote_state(remote_client_state)
        elif json_pose_dict is not None:
            self.from_json(json_pose_dict)

    def from_remote_state(
        self, remote_client_state: RemoteClientState
    ) -> None:
        """
        Read the XRPose from the provided RemoteClientState.
        """
        head_pos, head_rot = remote_client_state.get_head_pose(user_index=0)
        pos_left, rot_left = remote_client_state.get_hand_pose(
            user_index=0, hand_idx=0
        )
        pos_right, rot_right = remote_client_state.get_hand_pose(
            user_index=0, hand_idx=1
        )
        if None in [head_pos, pos_left, pos_right]:
            # one of the controllers or the headset has no valid state, this object is invalid
            self.valid = False
            return
        self.valid = True

        # collect the state info
        self.pos_head = head_pos
        self.rot_head = head_rot
        self.pos_left = pos_left
        self.rot_left = rot_left
        self.pos_right = pos_right
        self.rot_right = rot_right

    def from_json(self, json_pose_dict: Dict[str, List[float]]) -> None:
        """
        Set the XRPose from a JSON dictionary.
        """
        self.valid = True

        # collect the state info from the JSON dict
        self.pos_head = mn.Vector3(json_pose_dict["pos_head"])
        self.rot_head = mn.Quaternion(
            mn.Vector3(json_pose_dict["rot_head"][:3]),
            json_pose_dict["rot_head"][3],
        )
        self.pos_left = mn.Vector3(json_pose_dict["pos_left"])
        self.rot_left = mn.Quaternion(
            mn.Vector3(json_pose_dict["rot_left"][:3]),
            json_pose_dict["rot_left"][3],
        )
        self.pos_right = mn.Vector3(json_pose_dict["pos_right"])
        self.rot_right = mn.Quaternion(
            mn.Vector3(json_pose_dict["rot_right"][:3]),
            json_pose_dict["rot_right"][3],
        )

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Serializes the XRPose into a JSON compatible Dict.
        """
        if not self.valid:
            return None
        # extract the list formatted pose components
        json_dict = {
            "pos_head": list(self.pos_head),
            "rot_head": list(self.rot_head.vector) + [self.rot_head.scalar],
            "pos_left": list(self.pos_left),
            "rot_left": list(self.rot_left.vector) + [self.rot_left.scalar],
            "pos_right": list(self.pos_right),
            "rot_right": list(self.rot_right.vector) + [self.rot_right.scalar],
        }
        return json_dict

    def draw_pose(self, dblr: DebugLineRender, transform: mn.Matrix4 = None):
        """
        Draws the XRPose as axis frames for the head and hands.
        Applies the provided local to global transformation from the default XR coordinate space.
        """
        if transform is not None:
            dblr.push_transform(transform)

        head_transform = mn.Matrix4.from_(
            (self.rot_head).to_matrix(), self.pos_head
        )
        debug_draw_axis(dblr, head_transform, scale=0.5)

        left_control_t = mn.Matrix4.from_(
            (self.rot_left).to_matrix(), self.pos_left
        )
        debug_draw_axis(dblr, left_control_t, scale=0.25)

        right_control_t = mn.Matrix4.from_(
            (self.rot_right).to_matrix(), self.pos_right
        )
        debug_draw_axis(dblr, right_control_t, scale=0.25)

        if transform is not None:
            dblr.pop_transform()


class XRTrajectory:
    """
    A time series of XRPose states.
    """

    def __init__(self):
        self.traj: List[XRPose] = []

    def get_pose(self, ix: int):
        """
        Get the pose at the specified index.
        """
        return self.traj[ix]

    def add_pose(self, xr_pose: XRPose):
        """
        Add a pose to the trajectory.
        """
        self.traj.append(xr_pose)

    def save_json(self, filename: str = "xr_pose.json"):
        """
        Save the current trajectory as a JSON.
        """
        out_dir = os.path.dirname(filename)
        if len(out_dir) > 0:
            os.makedirs(out_dir, exist_ok=True)
        with open(filename, "w") as f:
            json_dict = {}
            for ix, pose in enumerate(self.traj):
                json_dict[ix] = pose.to_json_dict()
            json.dump(json_dict, f)

    def load_json(self, filename: str = "xr_pose.json"):
        """
        Load an XR trajectory from a JSON file.
        """
        if not os.path.exists(filename):
            print(
                f"Cannot load cached trajectory. Configured trajectory file {filename} does not exist."
            )
            return
        self.traj = []
        with open(filename, "r") as f:
            traj_dict = json.load(f)
            # JSON is a single object indexed sequentially by int keys
            for index in range(len(traj_dict)):
                self.traj.append(XRPose(json_pose_dict=traj_dict[str(index)]))


class XRPoseAdapter:
    """
    Encapsulates a set of utilities for adapting XR poses from between coordinate spaces.
    Includes helpers functions which make use of the adapter implicitly to reduce boilerplate code in the application.
    """

    def __init__(self) -> None:
        # this transform should be constructed to map xr local space into the final global coordinate space
        # NOTE: this is done within the application
        self.xr_local_to_global: mn.Matrix4 = mn.Matrix4()
        self.arm_extension_scale_factor = 1.75

    def xr_pose_transformed(
        self, xr_pose: XRPose, transform: mn.Matrix4
    ) -> XRPose:
        """
        Apply a transformation matrix to an XRPose and return the newly constructed object.
        """
        transformed_pose = XRPose()
        t_quat = mn.Quaternion.from_matrix(transform.rotation())
        transformed_pose.pos_head = transform.transform_point(xr_pose.pos_head)
        transformed_pose.pos_left = transform.transform_point(xr_pose.pos_left)
        transformed_pose.pos_right = transform.transform_point(
            xr_pose.pos_right
        )
        transformed_pose.rot_head = t_quat * xr_pose.rot_head
        transformed_pose.rot_left = t_quat * xr_pose.rot_left
        transformed_pose.rot_right = t_quat * xr_pose.rot_right
        transformed_pose.valid = True
        return transformed_pose

    def get_global_xr_pose(self, xr_pose: XRPose) -> XRPose:
        """
        Get a global space XRPose from a local space XRPose.
        Transform all elements of an XRPose with the local_to_global_transform and return the new object.
        """
        return self.extend_xr_reach(
            self.xr_pose_transformed(xr_pose, self.xr_local_to_global)
        )

    def extend_xr_reach(self, xr_pose: XRPose) -> XRPose:
        """
        Heuristic to extend the effective reach of the XR user by non-uniformly scaling the translation on the hand pose elements.
        """
        new_xr_pose = copy.deepcopy(xr_pose)
        head_to_left = xr_pose.pos_left - xr_pose.pos_head
        head_to_right = xr_pose.pos_right - xr_pose.pos_head
        new_xr_pose.pos_left = (
            xr_pose.pos_head + head_to_left * self.arm_extension_scale_factor
        )
        new_xr_pose.pos_right = (
            xr_pose.pos_head + head_to_right * self.arm_extension_scale_factor
        )
        return new_xr_pose

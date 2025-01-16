#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import pickle as pkl

import magnum as mn
import numpy as np

from habitat.articulated_agent_controllers import (
    HumanoidBaseController,
    Motion,
    Pose,
)
from habitat.articulated_agent_controllers.humanoid_base_controller import (
    BASE_HUMANOID_OFFSET,
)

MIN_ANGLE_TURN: float = 5.0  # If we turn less than this amount, we can just rotate the base and keep walking motion the same as if we had not rotated
TURNING_STEP_AMOUNT: float = (
    20.0  # The maximum angle we should be rotating at a given step
)
THRESHOLD_ROTATE_NOT_MOVE: float = 20.0  # The rotation angle above which we should only walk as if rotating in place
DIST_TO_STOP: float = (
    1e-9  # If the amount to move is this distance, just stop the character
)

from typing import List, Tuple


def bin_search(v, a, b, target_value):
    left = a
    right = b

    while left <= right:
        mid = (left + right) // 2

        if v[mid] == target_value:
            return mid
        elif v[mid] < target_value:
            left = mid + 1
        else:
            right = mid - 1

    return min(left, len(v) - 1)


class HumanoidRearrangeController(HumanoidBaseController):
    """
    Humanoid Controller, converts high level actions such as walk, or reach into joints positions
    """

    def __init__(
        self,
        walk_pose_path: str,
        motion_fps: int = 30,
        base_offset: mn.Vector3 = BASE_HUMANOID_OFFSET,
    ):
        """
        :param walk_pose_path: file containing the walking poses we care about.
        :param motion_fps: the 'frames per second' at which we should be advancing the pose.
        :base_offset: what is the offset between the root of the character and their feet.
        """

        self.obj_transform_base = mn.Matrix4()
        super().__init__(motion_fps, base_offset)

        self.min_angle_turn = MIN_ANGLE_TURN
        self.turning_step_amount = TURNING_STEP_AMOUNT
        self.threshold_rotate_not_move = THRESHOLD_ROTATE_NOT_MOVE

        if not os.path.isfile(walk_pose_path):
            raise RuntimeError(
                f"Path does {walk_pose_path} not exist. Reach out to the paper authors to obtain this data."
            )

        with open(walk_pose_path, "rb") as f:
            walk_data = pkl.load(f)
        walk_info = walk_data["walk_motion"]

        self.walk_motion = Motion(
            walk_info["joints_array"],
            walk_info["transform_array"],
            walk_info["displacement"],
            walk_info["fps"],
        )

        self.stop_pose = Pose(
            walk_data["stop_pose"]["joints"].reshape(-1),
            mn.Matrix4(walk_data["stop_pose"]["transform"]),
        )
        self.motion_fps = motion_fps
        self.dist_per_step_size = (
            self.walk_motion.displacement[-1] / self.walk_motion.num_poses
        )

        self.prev_orientation = None
        self.walk_mocap_frame = 0

        self.hand_processed_data = {}
        self._hand_names = ["left_hand", "right_hand"]
        ## Load hand data
        for hand_name in self._hand_names:
            if hand_name in walk_data:
                # Hand data contains two keys, pose_motion and coord_info
                # pose_motion: Contains information about each of the N poses.
                #   joints_array: list of joints represented as quaternions for
                #   each pose: (N * J) x 4.
                #   transform_array: an offset transform matrix for every pose
                # coord_info: dictionary specifying the bounds and number of bins used
                #   to compute the target poses
                hand_data = walk_data[hand_name]
                nposes = hand_data["pose_motion"]["transform_array"].shape[0]
                self.vpose_info = hand_data["coord_info"].item()
                hand_motion = Motion(
                    hand_data["pose_motion"]["joints_array"].reshape(
                        nposes, -1, 4
                    ),
                    hand_data["pose_motion"]["transform_array"],
                    None,
                    1,
                )
                self.hand_processed_data[hand_name] = self.build_ik_vectors(
                    hand_motion
                )
            else:
                self.hand_processed_data[hand_name] = None

    def set_framerate_for_linspeed(
        self, lin_speed: float, ang_speed: float, ctrl_freq: float
    ) -> None:
        """
        Set the linear and angular speed of the humanoid's root motion based on the relative framerate of the simulator and the motion trajectory.

        :param lin_speed: Speed of linear translation.
        :param ang_speed: Speed of the angular rotation (about Y-axis).
        :param ctrl_freq: Simulator step frequency. How many times does Simulator step to progress 1 second.
        """

        assert ctrl_freq > 0, "`ctrl_freq` must be strictly greater than zero."
        seconds_per_step = 1.0 / ctrl_freq
        meters_per_step = lin_speed * seconds_per_step
        frames_per_step = meters_per_step / self.dist_per_step_size
        self.motion_fps = self.walk_motion.fps / frames_per_step
        rotate_amount = ang_speed * seconds_per_step
        rotate_amount = rotate_amount * 180.0 / np.pi
        self.turning_step_amount = rotate_amount
        self.threshold_rotate_not_move = rotate_amount

    def calculate_stop_pose(self) -> None:
        """
        Calculates a stationary standing pose and sets it in the controller. Does not update the ManagedArticulatedObject state.
        """

        # the object transform does not change
        self.joint_pose = self.stop_pose.joints

    def calculate_turn_pose(self, target_position: mn.Vector3) -> None:
        """
        Calculates next frame of motion (joint positions + root rotation) to rotate the character towards the target position without translation and sets it in the controller.  Does not update the ManagedArticulatedObject state.
        """
        self.calculate_walk_pose(target_position, distance_multiplier=0)

    def calculate_walk_pose(
        self,
        target_position: mn.Vector3,
        distance_multiplier: float = 1.0,
    ) -> None:
        """
        Calculates next frame of a walking motion (joint positions + root rotation and translation), so that the humanoid moves toward the relative target_position. Does not update the ManagedArticulatedObject state.

        :param target_position: target position to move the character towards. Relative to the character's root translation.
        :param distance_multiplier: multiplier on translation speed. '== 0' creates a walk motion while not translating, for example to turn in place.
        """

        deg_per_rads = 180.0 / np.pi
        forward_V = target_position
        if (
            forward_V.length() < DIST_TO_STOP
            or np.isnan(target_position).any()
        ):
            self.calculate_stop_pose()
            return

        distance_to_walk = np.linalg.norm(forward_V)
        did_rotate = False

        # The angle we initially want to go to
        new_angle = np.arctan2(forward_V[0], forward_V[2]) * deg_per_rads
        if self.prev_orientation is not None:
            # If prev orientation is None, transition to this position directly
            prev_orientation = self.prev_orientation
            prev_angle = (
                np.arctan2(prev_orientation[0], prev_orientation[2])
                * deg_per_rads
            )

            forward_angle = new_angle - prev_angle
            forward_angle = (forward_angle + 180) % 360 - 180
            if np.abs(forward_angle) > self.min_angle_turn:
                actual_angle_move = self.turning_step_amount
                if abs(forward_angle) < actual_angle_move:
                    actual_angle_move = abs(forward_angle)
                new_angle = prev_angle + actual_angle_move * np.sign(
                    forward_angle
                )
                new_angle /= deg_per_rads
                did_rotate = True
            else:
                new_angle = new_angle / deg_per_rads

            forward_V = mn.Vector3(np.sin(new_angle), 0, np.cos(new_angle))

        forward_V = mn.Vector3(forward_V)
        forward_V = forward_V.normalized()
        self.prev_orientation = forward_V

        # Step size according to the FPS
        step_size = int(self.walk_motion.fps / self.motion_fps)

        if did_rotate:
            # When we rotate, we allow some movement
            distance_to_walk = self.dist_per_step_size * 2
            if np.abs(forward_angle) >= self.threshold_rotate_not_move:
                distance_to_walk *= 0

        # Step size according to how much we moved, this is so that
        # we don't overshoot if the speed of the character would it make
        # it move further than what `position` indicates
        step_size = max(
            1, min(step_size, int(distance_to_walk / self.dist_per_step_size))
        )

        if distance_multiplier == 0.0:
            step_size = 0

        # Advance mocap frame
        prev_mocap_frame = self.walk_mocap_frame
        self.walk_mocap_frame = (
            self.walk_mocap_frame + step_size
        ) % self.walk_motion.num_poses

        # Compute how much distance we covered in this motion
        prev_cum_distance_covered = self.walk_motion.displacement[
            prev_mocap_frame
        ]
        new_cum_distance_covered = self.walk_motion.displacement[
            self.walk_mocap_frame
        ]

        offset = 0
        if self.walk_mocap_frame < prev_mocap_frame:
            # We looped over the motion
            offset = self.walk_motion.displacement[-1]

        distance_covered = max(
            0, new_cum_distance_covered + offset - prev_cum_distance_covered
        )
        dist_diff = min(distance_to_walk, distance_covered)

        new_pose = self.walk_motion.poses[self.walk_mocap_frame]
        joint_pose, obj_transform = new_pose.joints, new_pose.root_transform

        # We correct the object transform

        forward_V_norm = mn.Vector3(
            [forward_V[2], forward_V[1], -forward_V[0]]
        )
        look_at_path_T = mn.Matrix4.look_at(
            self.obj_transform_base.translation,
            self.obj_transform_base.translation + forward_V_norm.normalized(),
            mn.Vector3.y_axis(),
        )

        # Remove the forward component, and orient according to forward_V
        add_rot = mn.Matrix4.rotation(mn.Rad(np.pi), mn.Vector3(0, 1, 0))
        obj_transform = add_rot @ obj_transform
        obj_transform.translation *= mn.Vector3.x_axis() + mn.Vector3.y_axis()

        # This is the rotation and translation caused by the current motion pose
        #  we still need to apply the base_transform to obtain the full transform
        self.obj_transform_offset = obj_transform

        # The base_transform here is independent of transforms caused by the current
        # motion pose.
        obj_transform_base = look_at_path_T
        forward_V_dist = forward_V * dist_diff * distance_multiplier
        obj_transform_base.translation += forward_V_dist

        rot_offset = mn.Matrix4.rotation(
            mn.Rad(-np.pi / 2), mn.Vector3(1, 0, 0)
        )
        self.obj_transform_base = obj_transform_base @ rot_offset
        self.joint_pose = joint_pose

    def translate_and_rotate_with_gait(
        self, forward_delta: float, rot_delta: float
    ):
        """
        Translates the character in the forward direction and rotates the character

        :param forward_delta: how much to move forward or backward
        :param rot_delta: how much to rotate (in radians)
        """
        forward_vec = self.obj_transform_base.transform_vector(
            mn.Vector3(1, 0, 0)
        )
        self.obj_transform_base.translation += forward_delta * forward_vec
        prev_angle = np.arctan2(forward_vec[0], forward_vec[2])
        new_angle = prev_angle + rot_delta
        new_forward_vec = mn.Vector3(np.cos(new_angle), 0, -np.sin(new_angle))
        look_at_path_T = mn.Matrix4.look_at(
            self.obj_transform_base.translation,
            self.obj_transform_base.translation + new_forward_vec.normalized(),
            mn.Vector3.y_axis(),
        )
        rot_offset = mn.Matrix4.rotation(
            mn.Rad(-np.pi / 2), mn.Vector3(1, 0, 0)
        )
        self.obj_transform_base = look_at_path_T @ rot_offset

        # Set a new position frame
        if forward_delta != 0:
            # Figure out which frame in the motion we should select
            current_frame = self.walk_mocap_frame
            target_displ = forward_delta
            total_displ = self.walk_motion.displacement[-1]
            abs_target_displ = np.abs(target_displ)

            # If the motion covers multiple cycles, take remainder
            if abs_target_displ > total_displ:
                num_cycles = int(target_displ / total_displ)
                abs_target_displ = abs_target_displ - total_displ * num_cycles
                target_displ = np.sign(target_displ) * abs_target_displ

            dist_to_end = (
                self.walk_motion.displacement[-1]
                - self.walk_motion.displacement[current_frame]
            )
            dist_to_beginning = self.walk_motion.displacement[current_frame]
            # Moving X backward is the same as moving Total_displ - X forward
            if np.sign(target_displ) < 0:
                target_displ += total_displ

            if dist_to_end < target_displ:
                target_displ -= dist_to_end
                a = 0
                b = current_frame
            else:
                a = current_frame
                b = len(self.walk_motion.displacement) - 1
                target_displ += dist_to_beginning
            # Bin search over which frame we should cover
            index = bin_search(
                self.walk_motion.displacement, a, b, target_displ
            )
            self.walk_mocap_frame = index

        new_pose = self.walk_motion.poses[self.walk_mocap_frame]
        joint_pose, obj_transform = new_pose.joints, new_pose.root_transform
        # Remove the forward component, and orient according to forward_V
        add_rot = mn.Matrix4.rotation(mn.Rad(np.pi), mn.Vector3(0, 1.0, 0))

        obj_transform = add_rot @ obj_transform
        obj_transform.translation *= mn.Vector3.x_axis() + mn.Vector3.y_axis()

        self.joint_pose = joint_pose
        self.obj_transform_offset = obj_transform

    def calculate_walk_pose_directional(
        self,
        target_position: mn.Vector3,
        distance_multiplier=1.0,
        target_dir: mn.Vector3 = None,
    ) -> None:
        """
        Calculates next frame of a walking motion (joint positions + root rotation and translation), so that the humanoid moves toward the relative target_position while facing the direction 'target_dir'. Does not update the ManagedArticulatedObject state.

        :param target_position: target position to move the character towards. Relative to the character's root translation.
        :param distance_multiplier: multiplier on translation speed. '== 0' creates a walk motion while not translating, for example to turn in place.
        :param target_dir: the relative position we should be looking at. If None, rotates the agent to face target_position. Otherwise, it moves the agent towards target_position but facing target_dir. This is important for moving backwards.
        """

        deg_per_rads = 180.0 / np.pi
        epsilon = 1e-5

        forward_V = target_position
        if forward_V.length() < epsilon or np.isnan(target_position).any():
            self.calculate_stop_pose()
            return
        distance_to_walk = float(np.linalg.norm(forward_V))
        did_rotate = False

        forward_V_orientation = forward_V
        # The angle we initially want to go to
        if target_dir is not None:
            new_angle = np.arctan2(target_dir[2], target_dir[0]) * deg_per_rads
            new_angle = (new_angle + 180) % 360 - 180
            if self.prev_orientation is not None:
                prev_angle = (
                    np.arctan2(
                        self.prev_orientation[2], self.prev_orientation[0]
                    )
                    * deg_per_rads
                )
            else:
                prev_angle = None

            new_angle_walk = (
                np.arctan2(forward_V[2], forward_V[0]) * deg_per_rads
            )

        else:
            new_angle = np.arctan2(forward_V[2], forward_V[0]) * deg_per_rads
            new_angle_walk = (
                np.arctan2(forward_V[2], forward_V[0]) * deg_per_rads
            )

        if self.prev_orientation is not None:
            # If prev orientation is None, transition to this position directly
            prev_orientation = self.prev_orientation
            prev_angle = (
                np.arctan2(prev_orientation[2], prev_orientation[0])
                * deg_per_rads
            )
            forward_angle = new_angle - prev_angle
            if forward_angle >= 180:
                forward_angle = forward_angle - 360
            if forward_angle <= -180:
                forward_angle = 360 + forward_angle

            if np.abs(forward_angle) > self.min_angle_turn:
                if target_dir is None:
                    actual_angle_move = self.turning_step_amount
                else:
                    actual_angle_move = self.turning_step_amount * 20
                if abs(forward_angle) < actual_angle_move:
                    actual_angle_move = abs(forward_angle)
                new_angle = prev_angle + actual_angle_move * np.sign(
                    forward_angle
                )
                new_angle /= deg_per_rads
                did_rotate = True
                new_angle_walk = new_angle
            else:
                new_angle = new_angle / deg_per_rads
                new_angle_walk = new_angle_walk / deg_per_rads
            forward_V = mn.Vector3(
                np.cos(new_angle_walk), 0, np.sin(new_angle_walk)
            )
            forward_V_orientation = mn.Vector3(
                np.cos(new_angle), 0, np.sin(new_angle)
            )

        forward_V = mn.Vector3(forward_V)
        forward_V = forward_V.normalized()
        self.prev_orientation = forward_V_orientation

        # TODO: Scale step size based on delta time.
        # step_size = int(self.walk_motion.fps / self.draw_fps)
        step_size = int(self.walk_motion.fps / 30.0)

        if did_rotate:
            # When we rotate, we allow some movement
            distance_to_walk = 0.05

        assert not np.isnan(
            distance_to_walk
        ), f"distance_to_walk is NaN: {distance_to_walk}"
        assert not np.isnan(
            self.dist_per_step_size
        ), f"distance_to_walk is NaN: {self.dist_per_step_size}"
        # Step size according to how much we moved, this is so that
        # we don't overshoot if the speed of the character would it make
        # it move further than what `position` indicates
        step_size = max(
            1, min(step_size, int(distance_to_walk / self.dist_per_step_size))
        )

        if distance_multiplier == 0.0:
            step_size = 0

        # Advance mocap frame
        prev_mocap_frame = self.walk_mocap_frame
        self.walk_mocap_frame = (
            self.walk_mocap_frame + step_size
        ) % self.walk_motion.num_poses

        # Compute how much distance we covered in this motion
        prev_cum_distance_covered = self.walk_motion.displacement[
            prev_mocap_frame
        ]
        new_cum_distance_covered = self.walk_motion.displacement[
            self.walk_mocap_frame
        ]

        offset = 0
        if self.walk_mocap_frame < prev_mocap_frame:
            # We looped over the motion
            offset = self.walk_motion.displacement[-1]

        distance_covered = max(
            0, new_cum_distance_covered + offset - prev_cum_distance_covered
        )
        dist_diff = min(distance_to_walk, distance_covered)

        new_pose = self.walk_motion.poses[self.walk_mocap_frame]
        joint_pose, obj_transform = new_pose.joints, new_pose.root_transform

        forward_V_norm = mn.Vector3(
            [
                forward_V_orientation[2],
                forward_V_orientation[1],
                -forward_V_orientation[0],
            ]
        )
        look_at_path_T = mn.Matrix4.look_at(
            self.obj_transform_base.translation,
            self.obj_transform_base.translation + forward_V_norm.normalized(),
            mn.Vector3.y_axis(),
        )

        # Remove the forward component, and orient according to forward_V
        add_rot = mn.Matrix4.rotation(mn.Rad(np.pi), mn.Vector3(0, 1.0, 0))

        obj_transform = add_rot @ obj_transform
        obj_transform.translation *= mn.Vector3.x_axis() + mn.Vector3.y_axis()

        # This is the rotation and translation caused by the current motion pose
        #  we still need to apply the base_transform to obtain the full transform
        self.obj_transform_offset = obj_transform

        # The base_transform here is independent of transforms caused by the current
        # motion pose.
        obj_transform_base = look_at_path_T
        forward_V_dist = forward_V * dist_diff * distance_multiplier
        obj_transform_base.translation += forward_V_dist

        rot_offset = mn.Matrix4.rotation(
            mn.Rad(-np.pi / 2), mn.Vector3(1, 0, 0)
        )
        self.obj_transform_base = obj_transform_base @ rot_offset
        self.joint_pose = joint_pose

    def build_ik_vectors(
        self, hand_motion: Motion
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Given a hand_motion Motion file containing different humanoid poses to reach objects with the hand, builds matrices of joint angles, root offset translations, and rotations so that they can be easily interpolated when reaching poses.

        :param hand_motion: The hand Motion object.

        :return: A Tuple of Lists containing joint quats, rotation quats, and translation vectors respectfully. Each list entry is a formatted matrix for the aforementioned values at a single Motion frame.
        """

        rotations, translations, joints = [], [], []
        for ind in range(len(hand_motion.poses)):
            curr_transform = mn.Matrix4(hand_motion.poses[ind].root_transform)

            quat_Rot = mn.Quaternion.from_matrix(curr_transform.rotation())
            joints.append(
                np.array(hand_motion.poses[ind].joints).reshape(-1, 4)[
                    None, ...
                ]
            )
            rotations.append(
                np.array(list(quat_Rot.vector) + [quat_Rot.scalar])[None, ...]
            )
            translations.append(
                np.array(curr_transform.translation)[None, ...]
            )

        add_rot = mn.Matrix4.rotation(mn.Rad(np.pi), mn.Vector3(0, 1.0, 0))

        obj_transform = add_rot @ self.walk_motion.poses[0].root_transform
        obj_transform.translation *= mn.Vector3.x_axis() + mn.Vector3.y_axis()
        curr_transform = obj_transform
        trans = (
            mn.Matrix4.rotation_y(mn.Rad(-np.pi / 2.0))
            @ mn.Matrix4.rotation_z(mn.Rad(-np.pi / 2.0))
        ).inverted()
        curr_transform = trans @ obj_transform

        quat_Rot = mn.Quaternion.from_matrix(curr_transform.rotation())
        joints.append(
            np.array(self.stop_pose.joints).reshape(-1, 4)[None, ...]
        )
        rotations.append(
            np.array(list(quat_Rot.vector) + [quat_Rot.scalar])[None, ...]
        )
        translations.append(np.array(curr_transform.translation)[None, ...])
        return (joints, rotations, translations)

    def _trilinear_interpolate_pose(
        self,
        position: mn.Vector3,
        hand_data: Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
    ) -> Tuple[List[np.ndarray], mn.Matrix4]:
        """
        Given a 3D coordinate position, computes the humanoid's joints states + root rotations and translations to reach that position using trilinear interpolation.

        :param position: The position to reach for.
        :param hand_data: The joint quats, root rotations, and root translation for each frame of the input reaching Motion.

        :return: joint state and root transform for the next step in the reaching sequence.
        """

        assert hand_data is not None

        joints, rotations, translations = hand_data

        def find_index_quant(
            minv: float,
            maxv: float,
            num_bins: int,
            value: float,
            interp: bool = False,
        ) -> Tuple[int, int, float]:
            # Find the quantization bins where a given value falls
            # E.g. if we have 3 points, min=0, max=1, value=0.75, it
            # will fall between 1 and 2
            if interp:
                value = max(min(value, maxv), 0)
            else:
                value = max(min(value, maxv), minv)
            value = min(value, maxv)
            value_norm = (value - minv) / (maxv - minv)

            index = value_norm * (num_bins - 1)

            lower = min(math.floor(index), num_bins - 1)
            upper = max(min(math.ceil(index), num_bins - 1), 0)
            value_norm_t = index - lower
            if lower < 0:
                min_poss_val = 0.0
                lower = int(
                    (min_poss_val - minv) * (num_bins - 1) / (maxv - minv)
                )
                value_norm_t = (index - lower) / -lower
                lower = -1

            return lower, upper, value_norm_t

        def comp_inter(x_i: int, y_i: int, z_i: int) -> int:
            # Given an integer index from 0 to num_bins - 1
            # on each dimension, compute the final index

            if y_i < 0 or x_i < 0 or z_i < 0:
                return -1
            index = (
                y_i
                * self.vpose_info["num_bins"][0]
                * self.vpose_info["num_bins"][2]
                + x_i * self.vpose_info["num_bins"][2]
                + z_i
            )
            return index

        def normalize_quat(quat_tens: np.ndarray) -> np.ndarray:
            # The last dimension contains the quaternion
            return quat_tens / np.linalg.norm(quat_tens, axis=-1)[..., None]

        def inter_data(
            x_i: Tuple[int, int, float],
            y_i: Tuple[int, int, float],
            z_i: Tuple[int, int, float],
            dat: np.ndarray,
            is_quat: bool = False,
        ) -> np.ndarray:
            """
            General trilinear interpolation function. Performs trilinear interpolation,
            normalizing the result if the values are represented as quaternions (is_quat)

            :param x_i, y_i, z_i: For the x,y,z dimensions, specifies the lower, upper, and normalized value
            so that we can perform interpolation in 3 dimensions
            :param dat: the values we want to interpolate.
            :param is_quat: used to normalize the value in case we are interpolating quaternions

            """
            x0, y0, z0 = x_i[0], y_i[0], z_i[0]
            x1, y1, z1 = x_i[1], y_i[1], z_i[1]
            xd, yd, zd = x_i[2], y_i[2], z_i[2]
            c000 = dat[comp_inter(x0, y0, z0)]
            c001 = dat[comp_inter(x0, y0, z1)]
            c010 = dat[comp_inter(x0, y1, z0)]
            c011 = dat[comp_inter(x0, y1, z1)]
            c100 = dat[comp_inter(x1, y0, z0)]
            c101 = dat[comp_inter(x1, y0, z1)]
            c110 = dat[comp_inter(x1, y1, z0)]
            c111 = dat[comp_inter(x1, y1, z1)]

            c00 = c000 * (1 - xd) + c100 * xd
            c01 = c001 * (1 - xd) + c101 * xd
            c10 = c010 * (1 - xd) + c110 * xd
            c11 = c011 * (1 - xd) + c111 * xd

            c0 = c00 * (1 - yd) + c10 * yd
            c1 = c01 * (1 - yd) + c11 * yd

            c = c0 * (1 - zd) + c1 * zd
            if is_quat:
                c = normalize_quat(c)

            return c

        relative_pos = position
        x_diff, y_diff, z_diff = relative_pos.x, relative_pos.y, relative_pos.z
        coord_diff = [x_diff, y_diff, z_diff]
        coord_data = [
            (
                self.vpose_info["min"][ind_diff],
                self.vpose_info["max"][ind_diff],
                self.vpose_info["num_bins"][ind_diff],
                coord_diff[ind_diff],
            )
            for ind_diff in range(3)
        ]
        # each value contains the lower, upper index and distance
        interp = [False, False, True]
        x_ind, y_ind, z_ind = [
            find_index_quant(*data, interp)
            for interp, data in zip(interp, coord_data)
        ]

        data_trans = np.concatenate(translations)
        data_rot = np.concatenate(rotations)
        data_joint = np.concatenate(joints)
        res_joint = inter_data(x_ind, y_ind, z_ind, data_joint, is_quat=True)
        res_trans = inter_data(x_ind, y_ind, z_ind, data_trans)
        res_rot = inter_data(x_ind, y_ind, z_ind, data_rot, is_quat=True)
        quat_rot = mn.Quaternion(mn.Vector3(res_rot[:3]), res_rot[-1])
        joint_list = list(res_joint.reshape(-1))
        transform = mn.Matrix4.from_(
            quat_rot.to_matrix(), mn.Vector3(res_trans)
        )
        return joint_list, transform

    def calculate_reach_pose(
        self, obj_pos: mn.Vector3, index_hand: int = 0
    ) -> None:
        """
        Updates the controller's humanoid state to reach position, obj_pos, with the hand.

        :param obj_pos: The position to reach with the hand
        :param index_hand: is 0 or 1 corresponding to the left or right hand
        """
        assert index_hand < 2
        hand_name = self._hand_names[index_hand]
        assert hand_name in self.hand_processed_data
        hand_data = self.hand_processed_data[hand_name]
        root_pos = self.obj_transform_base.translation
        inv_T = (
            mn.Matrix4.rotation_y(mn.Rad(-np.pi / 2.0))
            @ mn.Matrix4.rotation_x(mn.Rad(-np.pi / 2.0))
            @ self.obj_transform_base.inverted()
        )
        relative_pos = inv_T.transform_vector(obj_pos - root_pos)

        # TODO
        if hand_data is not None:
            curr_poses, curr_transform = self._trilinear_interpolate_pose(
                mn.Vector3(relative_pos), hand_data
            )

            self.obj_transform_offset = (
                mn.Matrix4.rotation_y(mn.Rad(-np.pi / 2.0))
                @ mn.Matrix4.rotation_z(mn.Rad(-np.pi / 2.0))
                @ curr_transform
            )

            self.joint_pose = curr_poses

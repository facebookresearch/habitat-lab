import math
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from scripts.expert_data.utils import rotation_conversions
from scripts.expert_data.utils.utils import create_T_matrix


class HeuristicOpenPolicy:
    def __init__(self, murp_env):
        self.murp_wrapper = murp_env
        self.env = murp_env.env
        self.TARGET_CONFIG = {
            "cabinet": [
                30,
                15,
                25,
                "_urdf_kitchen_FREMONT_KITCHENSET_FREMONT_KITCHENSET_CLEANED_urdf/kitchenset_cabinet",
            ],
            "shelf": [
                25,
                20,
                1,
                "_urdf_kitchen_FREMONT_KITCHENSET_FREMONT_KITCHENSET_CLEANED_urdf/kitchenset_door18",
            ],
            "island": [
                20,
                12,
                18,
                "_urdf_kitchen_FREMONT_KITCHENSET_FREMONT_KITCHENSET_CLEANED_urdf/kitchenset_island",
            ],
            "oven": [
                35,
                18,
                30,
                "_urdf_kitchen_FREMONT_KITCHENSET_FREMONT_KITCHENSET_CLEANED_urdf/kitchenset_ovendoor2",
            ],
            "fridge": [
                30,
                14,
                28,
                "_urdf_kitchen_FREMONT_KITCHENSET_FREMONT_KITCHENSET_CLEANED_urdf/kitchenset_fridgedoor1",
            ],
            "fridge_2": [
                30,
                20,
                19,
                "_urdf_kitchen_FREMONT_KITCHENSET_FREMONT_KITCHENSET_CLEANED_urdf/kitchenset_fridgedoor2",
            ],
            "freezer": [
                28,
                12,
                24,
                "_urdf_kitchen_FREMONT_KITCHENSET_FREMONT_KITCHENSET_CLEANED_urdf/kitchenset_freezer",
            ],
        }

    def set_targets(
        self, target_w_xyz, target_w_quat, target_joints, hand="left"
    ):
        self.target_w_xyz = target_w_xyz
        self.target_w_quat = target_w_quat
        target_quat = R.from_quat(target_w_quat, scalar_first=True)
        # Assumes (wxyz), isaac outputs (xyzw). If direct from isaac remember to swap.
        # self.target_w_rotmat = rotation_conversions.quaternion_to_matrix( self.target_w_quat )
        self.target_w_rotmat = target_quat.as_matrix()
        self.target_joints = target_joints

        # XYZ
        self.open_xyz = target_w_xyz.copy()
        if hand == "left":
            self.open_xyz[2] -= 0.1
            self.open_xyz[0] += 0.1

            # Pre-Grasp Targets
            OPEN_JOINTS = [1, 5, 9, 14]
            # OPEN_JOINTS = [0, 4, 6, 9]
            # Grasp fingers
            self.grasp_fingers = self.target_joints.copy()
            self.close_fingers = self.target_joints.copy()
            self.close_fingers[OPEN_JOINTS] += 0.2
        else:
            self.open_xyz[2] -= 0.1
            self.open_xyz[0] += 0.1
            SECONDARY_JOINTS = [2, 6, 10, 15]
            TERTIARY_JOINTS = [3, 7, 11]
            OPEN_JOINTS = [1, 5, 9]
            CURVE_JOINTS = [13]
            BASE_THUMB_JOINT = [12]
            self.grasp_fingers = self.target_joints.copy()
            self.close_fingers = self.target_joints.copy()
            self.close_fingers[BASE_THUMB_JOINT] += 1.1
            # self.close_fingers[CURVE_JOINTS] -=0.5
            self.close_fingers[SECONDARY_JOINTS] += 0.7
            self.close_fingers[TERTIARY_JOINTS] += 1.0
            self.close_fingers[OPEN_JOINTS] += 0.7

    def get_targets(self, name="target", hand="right"):
        # Lambda Changes
        if name == "target":
            return (
                torch.tensor(self.grasp_fingers, device="cuda:0"),
                torch.tensor(self.target_w_xyz, device="cuda:0"),
                torch.tensor(self.target_w_rotmat, device="cuda:0"),
            )

        elif name == "target_grip":
            return (
                torch.tensor(self.close_fingers),
                # torch.tensor(self.target_w_xyz),
                torch.tensor(self.murp_wrapper.get_curr_ee_pose()[0]),
                torch.tensor(self.target_w_rotmat),
            )

        elif name == "open":
            self.open_xyz = self.murp_wrapper.get_curr_ee_pose()[0]
            if hand == "right":
                # self.open_xyz[1] -= 0.1
                self.open_xyz[0] += 0.1

            return (
                torch.tensor(self.close_fingers, device="cuda:0"),
                torch.tensor(self.open_xyz, device="cuda:0"),
                torch.tensor(self.target_w_rotmat, device="cuda:0"),
            )

    def generate_action(self, cur_obs, name):
        cur_tar_wrist_xyz = cur_obs["wrist_tar_xyz"]
        cur_tar_fingers = cur_obs["tar_joints"][:16]

        # Phase 2 go to pick
        delta_t = 0.1
        self.step += 1

        tar_joints, tar_xyz, tar_rot = self.get_targets(name=name)

        delta_xyz = -(cur_tar_wrist_xyz - tar_xyz.to(cur_tar_wrist_xyz))
        print("cur_tar_wrist_xyz: ", cur_tar_wrist_xyz)
        print("tar_xyz: ", tar_xyz)
        print("delta_xyz: ", delta_xyz)
        delta_joints = tar_joints.to(cur_tar_fingers) - cur_tar_fingers

        tar_xyz = cur_tar_wrist_xyz + delta_xyz * delta_t
        tar_joint = cur_tar_fingers + delta_joints * delta_t
        # Orientation
        door_orientation = cur_obs["orientation_door"]
        door_orientation = torch.tensor(
            door_orientation, dtype=torch.float32, device="cuda:0"
        )  # Lambda Change
        door_rot = rotation_conversions.quaternion_to_matrix(door_orientation)

        rot_y = rotation_conversions.euler_angles_to_matrix(
            torch.tensor([math.pi, -math.pi, 0.0], device="cuda:0"), "XYZ"
        )
        target_rot = torch.einsum("ij,jk->ik", door_rot, rot_y)
        tar_rot = target_rot

        act = {
            "tar_xyz": tar_xyz.cpu().numpy(),
            "tar_rot": tar_rot.cpu().numpy(),
            "tar_fingers": tar_joint.cpu().numpy(),
        }

        return act

    def apply_rotation(self, quat_door):
        hab_T_door = R.from_quat(quat_door)
        isaac_T_hab_list = [-90, 0, 0]
        isaac_T_hab = R.from_euler("xyz", isaac_T_hab_list, degrees=True)
        isaac_T_door_mat = R.from_matrix(
            isaac_T_hab.as_matrix() @ hab_T_door.as_matrix()
        )
        isaac_T_door_quat = isaac_T_door_mat.as_quat()
        return isaac_T_door_quat

    def get_door_quat(self):
        path = self.TARGET_CONFIG[self.murp_wrapper.config.target_name][3]
        door_trans, door_orientation_quat = self.env.sim.get_prim_transform(
            os.path.join("/World/test_scene/", path), convention="quat"
        )
        self.murp_wrapper.visualize_pos(door_trans, "door")

        isaac_T_door_quat = self.apply_rotation(door_orientation_quat)
        print("door_quat: ", isaac_T_door_quat)
        return isaac_T_door_quat

    def grasp_obj(self, name):
        quat_door = self.get_door_quat()

        cur_obs = {
            "wrist_tar_xyz": torch.tensor(self.current_target_xyz),
            "tar_joints": torch.tensor(self.current_target_fingers),
            "orientation_door": torch.tensor(quat_door),
        }
        act = self.generate_action(cur_obs, name)

        base_T_hand = act["tar_xyz"]

        ee_pos, ee_rot = (
            self.env.sim.articulated_agent._robot_wrapper.hand_pose()
        )
        base_T_ee = create_T_matrix(ee_pos, ee_rot)

        hand_pos, hand_rot = (
            self.env.sim.articulated_agent._robot_wrapper.hand_pose()
        )
        base_T_hand = create_T_matrix(hand_pos, hand_rot)

        ee_T_hand = np.linalg.inv(base_T_ee) @ base_T_hand
        base_T_hand = base_T_ee @ ee_T_hand
        print("base_T_hand: ", base_T_hand[:, -1])

        self.murp_wrapper.visualize_pos(act["tar_xyz"], "hand")
        # self.move_hand_joints(act["tar_fingers"], timeout=10)
        target_rot_mat = R.from_matrix(act["tar_rot"])
        target_rot_rpy = target_rot_mat.as_euler("xyz", degrees=False)

        print(
            f"Policy XYZ {act['tar_xyz']},  Rot {np.rad2deg(target_rot_rpy)}"
        )
        curr_xyz, curr_ori = self.murp_wrapper.get_curr_ee_pose()
        print(f"Curr XYZ {curr_xyz}, Rot {curr_ori}")
        target_rot_rpy = self.target_ee_rot
        # if name == "open" and self.step > 10:
        #     self.move_base_ee_and_hand(
        #         -0.1,
        #         0.0,
        #         act["tar_xyz"],
        #         target_rot_rpy,
        #         act["tar_fingers"],
        #         timeout=10,
        #     )
        #     # self.move_base(
        #     #     -0.1,
        #     #     0.0,
        #     # )
        # else:
        self.murp_wrapper.move_ee_and_hand(
            act["tar_xyz"], target_rot_rpy, act["tar_fingers"], timeout=10
        )
        self.current_target_fingers = act["tar_fingers"]
        self.current_target_xyz = act["tar_xyz"]
        _current_target_rotmat = R.from_matrix(act["tar_rot"])
        self.current_target_quat = _current_target_rotmat.as_quat()

    def execute_grasp_sequence(
        self, hand, grip_iters, open_iters, move_iters=None
    ):
        self.target_ee_pos, self.target_ee_rot = (
            self.env.current_episode.action_target[1],
            self.env.current_episode.action_target[2],
        )

        self.murp_wrapper.move_to_ee(
            self.target_ee_pos,
            self.target_ee_rot,
            grasp="pre_grasp" if hand == "left" else "open",
            timeout=300 if hand == "left" else 200,
        )

        self.step = 0
        self.current_target_fingers = (
            self.env.sim.articulated_agent._robot_wrapper.right_hand_joint_pos
        )
        self.current_target_xyz = self.target_ee_pos
        target_xyz, target_ori = self.murp_wrapper.get_curr_ee_pose()
        target_xyz[1] -= 0.52
        target_ori_rpy = R.from_euler("xyz", target_ori, degrees=True)
        target_quaternion = target_ori_rpy.as_quat(scalar_first=True)  # wxzy
        target_joints = (
            self.env.sim.articulated_agent._robot_wrapper.right_hand_joint_pos
        )
        self.set_targets(
            target_w_xyz=target_xyz,
            target_w_quat=target_quaternion,
            target_joints=target_joints,
            hand=hand,
        )
        if move_iters:
            for _ in range(move_iters):
                self.murp_wrapper.move_base(1.0, 0.0)

        # Grasp and open object
        for _ in range(grip_iters):
            self.grasp_obj(name="target_grip")

        for _ in range(open_iters):
            self.grasp_obj(name="open")

        # Move robot back
        for _ in range(10):
            self.murp_wrapper.move_base(-1.0, 0.0)

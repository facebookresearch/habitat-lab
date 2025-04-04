import math
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from scripts.expert_data.utils import rotation_conversions
from scripts.expert_data.utils.utils import create_T_matrix


class HeuristicPickPolicy:
    def __init__(self, murp_env):
        self.murp_wrapper = murp_env
        self.env = murp_env.env
        self.config = murp_env.config
        self.grasp_cache_path = self.config.grasp_cache_path
        self.TARGET_CONFIG = {
            "living_room_console": [
                40,
                10,
                7,
                " ",
            ],
            "shelf": [
                25,
                20,
                5,
                " ",
            ],
            "island": [
                20,
                12,
                5,
                " ",
            ],
            "sofa": [
                35,
                18,
                5,
                " ",
            ],
            "kitchen_set": [
                30,
                14,
                5,
                " ",
            ],
            "office_room_console": [
                30,
                20,
                5,
                " ",
            ],
            "bedroom_room_console": [
                28,
                12,
                5,
                " ",
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
        target_w_xyz = np.array(target_w_xyz)
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
            SECONDARY_JOINTS = [2, 6, 10, 15]
            TERTIARY_JOINTS = [3, 7, 11]
            OPEN_JOINTS = [1, 5, 9]
            CURVE_JOINTS = [13]
            BASE_THUMB_JOINT = [12]
            self.grasp_fingers = self.target_joints.copy()
            self.grasp_fingers[OPEN_JOINTS] -=0.6
            self.close_fingers = self.target_joints.copy()
            # self.close_fingers[CURVE_JOINTS] -=0.5
            self.close_fingers[OPEN_JOINTS] += 0.8

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
                torch.tensor(self.grasp_fingers),
                torch.tensor(self.target_w_xyz),
                torch.tensor(self.target_w_rotmat),
            )

        elif name == "open":
            self.open_xyz = self.murp_wrapper.get_curr_ee_pose()[0]
            if hand == "right":
                self.open_xyz[1] += 0.5
                # self.open_xyz[0] += 0.1
            return (
                torch.tensor(self.close_fingers),
                torch.tensor(self.open_xyz),
                torch.tensor(self.target_w_rotmat),
            )

    def generate_action_pick(self, cur_obs, name):
        cur_tar_wrist_xyz = cur_obs["wrist_tar_xyz"]
        cur_tar_wrist_quat = cur_obs["wrist_tar_quat"]
        cur_tar_wrist_rotmat = rotation_conversions.quaternion_to_matrix(cur_tar_wrist_quat)
        cur_tar_fingers =  cur_obs["tar_joints"][:16]

        # Phase 2 go to pick
        delta_t = 0.1
        self.step += 1
        print(self.step)

        tar_joints, tar_xyz, tar_rot = self.get_targets(name=name)

        delta_xyz = -(cur_tar_wrist_xyz - tar_xyz.to(cur_tar_wrist_xyz))
        delta_xyz = torch.clamp(delta_xyz, -0.1, 0.1)
        print("cur_tar_wrist_xyz: ", cur_tar_wrist_xyz)
        print("tar_xyz: ", tar_xyz)
        print("delta_xyz: ", delta_xyz)
        # Wrist Rot
        # delta_rot = torch.einsum('bij,bjk->bik',cur_tar_wrist_rotmat.transpose(-1,-2), tar_rot.to(cur_tar_wrist_xyz))
        cur_tar_wrist_rotmat = cur_tar_wrist_rotmat.to(dtype=torch.float32)
        tar_rot = tar_rot.to(dtype=torch.float32)
        delta_rot = torch.einsum('ij,jk->ik', cur_tar_wrist_rotmat.transpose(-1, -2), tar_rot.to(dtype=torch.float32))


        delta_ax = rotation_conversions.matrix_to_axis_angle(delta_rot)
        #delta_ax = torch.clamp(delta_ax, -1., 1.)
        # Fingers
        delta_joints = -(cur_tar_fingers - tar_joints.to(cur_tar_wrist_xyz))

        act = {
            "delta_wrist_xyz": delta_xyz.cpu().numpy(),
            "delta_ax_ang": delta_ax.cpu().numpy(),
            "delta_joints": delta_joints.cpu().numpy(),
        }

        return act

    def grasp_object(self, name):
        cur_obs = {
            "wrist_tar_xyz": torch.tensor(self.current_target_xyz),
            "tar_joints": torch.tensor(self.current_target_fingers),
            "wrist_tar_quat": torch.tensor(self.current_target_rot)
        }
        act = self.generate_action_pick(cur_obs, name)
        dt = 0.01
        self.murp_wrapper.visualize_pos(act["delta_wrist_xyz"], "hand")
        axis_angle = act["delta_ax_ang"]
        target_rot_mat = R.from_rotvec(axis_angle)
        target_rot_rpy = target_rot_mat.as_euler("xyz", degrees=False)

        curr_xyz, curr_ori = self.murp_wrapper.get_curr_ee_pose()
        print(f"Curr XYZ {curr_xyz}, Rot {curr_ori}")
        target_rot_rpy = self.target_ee_rot
        print("DELTA ROTPY",self.target_ee_rot - target_rot_rpy)
        new_target= self.current_target_xyz + act["delta_wrist_xyz"]
        new_joints = self.current_target_fingers + act["delta_joints"]
        self.murp_wrapper.move_ee_and_hand(
            new_target, target_rot_rpy, new_joints, timeout=10 , text=name
        )
        self.current_target_fingers += act["delta_joints"] #updating joints
        self.current_target_xyz += act["delta_wrist_xyz"]*0.1 #updating wrist xyz
        _current_target_rotmat = R.from_quat(self.current_target_rot).as_matrix()
        _current_target_rotmat = torch.Tensor(_current_target_rotmat)
        _delta_rot = rotation_conversions.axis_angle_to_matrix(act['delta_ax_ang']*dt)
        _current_target_rotmat = torch.einsum('ij,jk->ik',_current_target_rotmat, _delta_rot)
        self.current_target_rot = R.from_matrix(_current_target_rotmat.numpy()).as_quat() #updating wrist quat
    
    def map_joints(self, joints, from_isaac=True):
        # map the joints from isaac convention to habitat convention
        # habitat convention is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        # isaac convention is [ 0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11]
        # map the joints from habitat convention to isaac convention
        if from_isaac:
            joints_hab = np.zeros(16)
            joints_hab[:4] = joints[:4]
            joints_hab[4:8] = joints[8:12]
            joints_hab[8:12] = joints[12:]
            joints_hab[12:] = joints[4:8]
            return joints_hab
        else:
            joints_isaac = np.zeros(16)
            joints_isaac[:4] = joints[:4]
            joints_isaac[4:8] = joints[8:12]
            joints_isaac[8:12] = joints[12:]
            joints_isaac[12:] = joints[4:8]
            return joints_isaac
   
    def acquire_info(self,data_cache,case):
            scene = data_cache[case]
            obj_name = case.split('_')[0]
            robot_dof = scene['robot_dof_pos']
            robot_base = scene['robot_root_state']
            robot_xyz = robot_base[:, :3]
            robot_quat = robot_base[:, 3:7]

            obj_state = scene['object_root_state']
            obj_xyz = obj_state[:, :3]
            obj_quat = obj_state[:, 3:7]

            return obj_name, robot_dof, {'xyz': robot_xyz, 'quat':robot_quat}, {'xyz': obj_xyz, 'quat':obj_quat}
    def execute_grasp_sequence(
        self, hand, grip_iters, open_iters, move_iters=None
    ):
        target = self.env.sim._rigid_objects[0].transformation
        rotation_matrix = np.array([
            [target[0].x, target[0].y, target[0].z],
            [target[1].x, target[1].y, target[1].z],
            [target[2].x, target[2].y, target[2].z]
        ])
        # rpy = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
        # perpendicular_rpy = rpy + np.array([90,90,0])
        # print("RPY",perpendicular_rpy)
        rpy = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
        diffs= np.array([-1.76995723,89.79812903,-89.99673813]) - rpy
        perpendicular_rpy = rpy + diffs

        self.target_ee_pos, self.target_ee_rot = (
            self.env.sim._rigid_objects[0].translation,
            perpendicular_rpy,
        )
        self.target_ee_pos[1]+=0.20
        self.murp_wrapper.visualize_pos( self.target_ee_pos, "ee")


        self.murp_wrapper.move_to_ee(
            self.target_ee_pos,
            self.target_ee_rot,
            grasp="pre_grasp" if hand == "left" else "open",
            timeout=300 if hand == "left" else 300,
        )


        self.step = 0
        self.current_target_fingers = (
            self.env.sim.articulated_agent._robot_wrapper.right_hand_joint_pos
        )
        load_file = self.grasp_cache_path
        data = torch.load(load_file, map_location='cpu')
        data_cache = data['cache']
        cases = list(data_cache.keys())
        obj_name, robot_dof, robot_pose, obj_pose = self.acquire_info(data_cache,cases[1])
        robot_dof=self.map_joints(self.env.sim.target_joints)
        # self.current_target_xyz = self.target_ee_pos
        tar_rot=R.from_euler("xyz", self.target_ee_rot, degrees=True)
        tar_quat = tar_rot.as_quat(scalar_first=True)
        self.current_target_rot = R.from_euler('xyz', self.target_ee_rot, degrees=True).as_quat()
        target_xyz, target_ori = self.murp_wrapper.get_curr_ee_pose()
        self.current_target_xyz = target_xyz
        target_ori_rpy = R.from_euler("xyz", target_ori, degrees=True)
        target_quaternion = target_ori_rpy.as_quat(scalar_first=True)  # wxzy
        target_joints = (
            self.env.sim.articulated_agent._robot_wrapper.right_hand_joint_pos
        )
        target_xyz = self.target_ee_pos
        target_xyz[1]-=0.05
        self.set_targets(
            target_w_xyz=target_xyz,
            target_w_quat=tar_quat,
            target_joints=robot_dof,
            hand=hand,
        )
        if move_iters:
            for _ in range(move_iters):
                self.murp_wrapper.move_base(1.0, 0.0)

        # Grasp and open object
        for _ in range(grip_iters):
            self.grasp_object(name="target_grip")

        for _ in range(open_iters):
            self.grasp_object(name="open")

        # Move robot back
        for _ in range(10):
            self.murp_wrapper.move_base(-1.0, 0.0)

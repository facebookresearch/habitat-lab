# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np
import numpy.typing as npt
from mochi_gym.envs.allegro_in_hand_env import (
    AllegroInHandEnv,
    AllegroInHandEnvCfg,
)
from mochi_gym.envs.gym_base_env_cfg import GymBaseEnvCfg
from mochi_gym.envs.mochi_env import MochiEnv
from mochi_gym.envs.mochi_env_cfg import RenderMode

from habitat.mochi.mochi_utils import habitat_to_mochi_position, quat_to_rotvec
from habitat.mochi.mochi_visualizer import MochiVisualizer

# from habitat.mochi.mochi_debug_drawer import MochiDebugDrawer

import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np

class FrankaTeleop:
    def __init__(self, lam=0.1):
        self.robot = rtb.models.ETS.Panda()
        self.lam = lam  # damping factor for stability

        # hand-tuned to match neutral hand pose
        axis = np.array([0.75, 1.5, 0.25])
        axis = axis / np.linalg.norm(axis)   # normalize
        self.R_neutral = sm.SO3.AngVec(np.pi/2, axis)

        self._position_clamp_min = np.array([0.2, -0.5, 0.0])
        self._position_clamp_max = np.array([0.7,  0.5, 0.7])

    def SE3_from_pos_rotvec(pos, rotvec):
        pos = np.asarray(pos, float)
        theta = float(np.linalg.norm(rotvec))
        if theta < 1e-8:
            R = sm.SO3()
        else:
            axis = np.asarray(rotvec, float) / theta
            R = sm.SO3.AngVec(theta, axis)
        # World pose with world translation `pos`
        T = sm.SE3.Trans(pos) * sm.SE3(R)   # => [R, pos]
        # sanity check during bring-up:
        # assert np.allclose(T.t, pos)
        return T

    def step(self, q, target, dt=0.05, gain_pos=1.0, gain_rot=1.0):

        pos, rotvec = target
        T_d = FrankaTeleop.SE3_from_pos_rotvec(pos, rotvec)

        # Current pose
        T = self.robot.fkine(q)
        R = T.R

        # --- Errors in CONSISTENT (body) frame ---
        # linear error to body
        e_p_b = R.T @ (T_d.t - T.t)

        # quaternion orientation error (already body)
        q_cur = sm.UnitQuaternion(R)
        q_des = sm.UnitQuaternion(T_d.R)

        q_err = q_cur.inv() * q_des
        e_o = q_err.v.copy()   # 3-vector part
        if q_err.s < 0:
            e_o = -e_o

        # scale position vs orientation
        v_b = np.hstack((gain_pos * e_p_b, gain_rot * e_o))  # shape (6,)

        # clamp cartesian speeds (optional)
        v_lin_max, v_ang_max = 0.25, 1.0  # m/s, rad/s
        v_b[:3]  = np.clip(v_b[:3], -v_lin_max, v_lin_max)
        v_b[3:]  = np.clip(v_b[3:], -v_ang_max, v_ang_max)

        # Jacobian in body frame
        J = self.robot.jacobe(q)  # 6x7

        # Damped least squares using solve (stable, no explicit inv)
        A = J @ J.T + self.lam**2 * np.eye(6)
        J_pinv = J.T @ np.linalg.solve(A, np.eye(6))

        dq = J_pinv @ v_b  # 7,
        return q + dq * dt

    def step_until_progress(self, q0, target,
                            min_pos_step=0.02,             # meters (X cm)
                            min_rot_step_deg=10.0,         # degrees (Y)
                            max_iters=20,                  # (Z)
                            dt=0.05, gain_pos=1.0, gain_rot=1.0):
        """
        Repeatedly call self.step(...) until we've advanced at least:
        - `min_pos_step` along the straight-line direction to the target position, OR
        - `min_rot_step_deg` reduction in geodesic angle to target orientation,
        or we hit `max_iters`.

        Progress is monotone: moving away doesn't count.
        """
        pos_goal, rotvec_goal = target
        T_goal = FrankaTeleop.SE3_from_pos_rotvec(pos_goal, rotvec_goal)

        q = q0.copy()

        # Starting EE pose
        T0 = self.robot.fkine(q0)
        pos0 = T0.t
        R0   = T0.R

        # Direction to target position (for signed progress)
        d_pos = pos_goal - pos0
        dist_goal = float(np.linalg.norm(d_pos))
        dir_to_goal = d_pos / dist_goal if dist_goal > 1e-12 else np.zeros(3)

        # Initial angular error to goal (geodesic on SO(3))
        theta_start, _ = sm.SO3(R0.T @ T_goal.R).angvec()

        # Thresholds
        min_rot_step = np.deg2rad(min_rot_step_deg)

        for _ in range(max_iters):
            # One normal resolved-rate step (the one that already "works great")
            q = self.step(q, target, dt=dt, gain_pos=gain_pos, gain_rot=gain_rot)

            # Measure positional progress along the correct direction
            T = self.robot.fkine(q)
            pos = T.t
            s = float(np.dot(pos - pos0, dir_to_goal)) if dist_goal > 1e-12 else 0.0
            pos_progress = max(0.0, min(s, dist_goal))   # no backward or beyond-goal credit

            # Measure rotational progress as reduction in angle-to-goal
            theta_now, _ = sm.SO3(T.R.T @ T_goal.R).angvec()
            rot_progress = max(0.0, float(theta_start - theta_now))

            if pos_progress >= min_pos_step or rot_progress >= min_rot_step:
                break

        return q

    def ee_pose(self, q):
        """
        Return (pos, rotvec) for a joint vector.
        rotvec is a 3-vector (axis * angle), with norm = angle (radians).
        """
        T = self.robot.fkine(q)
        pos = T.t

        θ, u = sm.SO3(T.R).angvec()   # angle, axis
        rotvec = θ * u

        return pos, rotvec

    def quat_xyzw_to_rpy(self, rotation_xyzw):
        """
        Convert quaternion to roll-pitch-yaw (radians).
        
        Args:
            q : iterable (x, y, z, w) quaternion
        
        Returns:
            (roll, pitch, yaw)
        """
        x, y, z, w = rotation_xyzw

        # Normalize to be safe
        norm = np.linalg.norm([x, y, z, w])
        if norm == 0:
            raise ValueError("Zero-length quaternion")
        x, y, z, w = x/norm, y/norm, z/norm, w/norm

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w*x + y*z)
        cosr_cosp = 1 - 2 * (x*x + y*y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w*y - z*x)
        if abs(sinp) >= 1:
            pitch = np.pi/2 * np.sign(sinp)  # clamp
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w*z + x*y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def get_position_clamp_range(self):
        return self._position_clamp_min, self._position_clamp_max

    def clamp_target(self, ee_pose):
        """
        Clamp EE target to safe workspace and orientation, around a neutral rotation.
        """
        assert isinstance(ee_pose, (list, tuple)) and len(ee_pose) == 2
        pos, rotvec = ee_pose

        clamp_min, clamp_max = self.get_position_clamp_range()
        rotvec_max_angle = np.deg2rad(150)  # ~2.62 rad

        # position box clamp
        pos_clamped = np.array([
            np.clip(pos[0], clamp_min[0], clamp_max[0]),
            np.clip(pos[1], clamp_min[1], clamp_max[1]),
            np.clip(pos[2], clamp_min[2], clamp_max[2])
        ])

        # todo: reenable clamping once we have real hand-tracking
        if False:
            # orientation clamp: relative to neutral
            rotvec = np.asarray(rotvec, float)
            theta = np.linalg.norm(rotvec)
            R_target = sm.SO3() if theta < 1e-8 else sm.SO3.AngVec(theta, rotvec/theta)

            R_err = self.R_neutral.T @ R_target
            θ, u = R_err.angvec()

            if θ > rotvec_max_angle:
                R_err = sm.SO3.AngVec(rotvec_max_angle, u)

            R_clamped = self.R_neutral @ R_err
            θc, uc = R_clamped.angvec()
            rotvec_clamped = θc * uc
        else:
            rotvec_clamped = rotvec

        return pos_clamped, rotvec_clamped    

def get_murp_joint_to_dof_index_map():
    return {
        "left_fr3_joint1": 0,
        "left_fr3_joint2": 2,
        "left_fr3_joint3": 4,
        "left_fr3_joint4": 6,
        "left_fr3_joint5": 8,
        "left_fr3_joint6": 10,
        "left_fr3_joint7": 12,
        "right_fr3_joint1": 1,
        "right_fr3_joint2": 3,
        "right_fr3_joint3": 5,
        "right_fr3_joint4": 7,
        "right_fr3_joint5": 9,
        "right_fr3_joint6": 11,
        "right_fr3_joint7": 13,
    }    
    # this is the order from the URDF but it is wrong
    # return {
    #     "left_fr3_joint1": 0,
    #     "left_fr3_joint2": 1,
    #     "left_fr3_joint3": 2,
    #     "left_fr3_joint4": 3,
    #     "left_fr3_joint5": 4,
    #     "left_fr3_joint6": 5,
    #     "left_fr3_joint7": 6,
    #     "right_fr3_joint1": 7,
    #     "right_fr3_joint2": 8,
    #     "right_fr3_joint3": 9,
    #     "right_fr3_joint4": 10,
    #     "right_fr3_joint5": 11,
    #     "right_fr3_joint6": 12,
    #     "right_fr3_joint7": 13,
    #     "bogie_left_joint": 14,
    #     "bogie_right_joint": 15,
    #     "joint_0.0": 16,
    #     "joint_1.0": 17,
    #     "joint_2.0": 18,
    #     "joint_3.0": 19,
    #     "joint_4.0": 20,
    #     "joint_5.0": 21,
    #     "joint_6.0": 22,
    #     "joint_7.0": 23,
    #     "joint_8.0": 24,
    #     "joint_9.0": 25,
    #     "joint_10.0": 26,
    #     "joint_11.0": 27,
    #     "joint_12.0": 28,
    #     "joint_13.0": 29,
    #     "joint_14.0": 30,
    #     "joint_15.0": 31,
    #     "joint_l_0.0": 32,
    #     "joint_l_1.0": 33,
    #     "joint_l_2.0": 34,
    #     "joint_l_3.0": 35,
    #     "joint_l_4.0": 36,
    #     "joint_l_5.0": 37,
    #     "joint_l_6.0": 38,
    #     "joint_l_7.0": 39,
    #     "joint_l_8.0": 40,
    #     "joint_l_9.0": 41,
    #     "joint_l_10.0": 42,
    #     "joint_l_11.0": 43,
    #     "joint_l_12.0": 44,
    #     "joint_l_13.0": 45,
    #     "joint_l_14.0": 46,
    #     "joint_l_15.0": 47,
    # }
    


class MochiWrapper:
    def __init__(self, hab_sim, do_render=False):
        environment = AllegroInHandEnv(
            AllegroInHandEnvCfg(
                steps_per_episode=-1,
                render_mode=RenderMode.HUMAN if do_render else RenderMode.NONE,
                simulation_frequency=100,
                control_frequency=100,
                do_rgb=False,
                worker_index=0,
                # require for our position controller to work
                force_identity_agent_pose=True,
            )
        )

        self._env = environment

        self._env.reset()

        self._object_handles = {}
        names = self._env._mochi.get_actors_names()

        for name in names:
            handle = self._env._mochi.get_actor_handle(name)
            self._object_handles[name] = handle

        self._mochi_visualizer = MochiVisualizer(hab_sim, self._env._mochi)

        self._mochi_visualizer.add_render_map(
            "./data/mochi_vr_data/test_scene.render_map.json"
        )

        NUM_FRANKA_JOINTS = 7
        num_base_dof = 6
        dof_map = get_murp_joint_to_dof_index_map()
        self._franka_arm_dof_idxs = [
            [dof_map[f"{arm}_fr3_joint{j}"] + num_base_dof for j in range(1, NUM_FRANKA_JOINTS + 1)]
            for arm in ["left", "right"]
        ]

        self._franka_ik_helpers = [FrankaTeleop(), FrankaTeleop()]

        temp = 0

        # render_actors =
        # self._debug_drawer = MochiDebugDrawer(


    def _update_metahand_and_get_action(self, dummy_metahand):
        target_pose = self._env._previous_target_pose
        target_pose[0:3] = list(
            habitat_to_mochi_position(dummy_metahand.target_base_position)
        )
        target_pose[3:6] = quat_to_rotvec(dummy_metahand.target_base_rotation)

        # note different finger convention

        # source
        # 0 pointer twist
        # 1 thumb rotate
        # 2 ring twist
        # 3 pinky twist
        # 4 pointer base
        # 5 thumb twist
        # 6 ring base
        # 7 pinky base
        # 8 pointer mid
        # 9 thumb base
        # 10 ring mid
        # 11 pinky mid
        # 12 pointer tip
        # 13 thumb tip
        # 14 ring tip
        # 15 pinky tip

        # dest
        # 0..4 finger twist and thumb rotate
        # 4..8 finger first joint bend and thumb twist
        # 8..12 finger second joint bend and thumb first joint bend
        # 12..16 last joint bend

        # 0,4,8,12 -> thumb rotate, thumb twist, then bend
        # 1,5,9,13 -> pinky twist and joint bends
        # 2,6,10,14 -> ring twist and joint bends
        # 3,7,11,15 -> pointer twist and joint bends

        remap = {
            0: 3,
            1: 0,
            2: 2,
            3: 1,
            4: 7,
            5: 4,
            6: 6,
            7: 5,
            8: 11,
            9: 8,
            10: 10,
            11: 9,
            12: 15,
            13: 12,
            14: 14,
            15: 13,
        }

        start = 6
        for src in remap:
            dest = remap[src]
            target_pose[start + dest] = dummy_metahand._target_joint_positions[
                src
            ]

        # target_pose[6:22] = [0.0] * 16  # temp

        # target_pose[6:22] = dummy_metahand._target_joint_positions

        action = [0.0] * 16
        return action

    def step(self, dummy_metahand):
        env = self._env

        action = self._update_metahand_and_get_action(dummy_metahand)
        # action = [0.0] * 16

        # Send the computed action to the env.
        _, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset()

    def pre_render(self):
        self._mochi_visualizer.flush_to_hab_sim()

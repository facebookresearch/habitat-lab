#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from PIL import Image

from habitat.tasks.rearrange.utils import IkHelper
from habitat_baselines.motion_planning.mp_sim import MpSim
from habitat_baselines.motion_planning.mp_spaces import MpSpace
from habitat_baselines.motion_planning.robot_target import (
    ObjectGraspTarget,
    RobotTarget,
)


class GraspGenerator:
    def __init__(
        self,
        use_sim: MpSim,
        mp_space: MpSpace,
        ik: IkHelper,
        mp,
        should_render: bool,
        grasp_thresh: float,
        n_gen_grasps: int,
        knows_other_objs: bool,
        log_dir: str,
        is_verbose: bool,
    ):
        self._mp_sim = use_sim
        self._ik = ik
        self._mp_space = mp_space
        (
            self._lower_joint_lims,
            self._upper_joint_lims,
        ) = self._ik.get_joint_limits()
        self.mp = mp
        self._should_render = should_render
        self._grasp_thresh = grasp_thresh
        self._n_gen_grasps = n_gen_grasps
        self.knows_other_objs = knows_other_objs
        self._log_dir = log_dir
        self._is_verbose = is_verbose

    def get_def_js(self):
        # A reference state which we should generally stay close to.
        return np.array([-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005])

    def get_targ_obj(self, start_js, obj_id):
        pass

    def _gen_goal_state(self, local_ee_targ, grasp_idx=0, timeout=100):
        """
        - local_ee_targ: 3D desired EE position in robot's base coordinate frame.
        - grasp_idx: The grasp index attempt. Used for debugging.
        Returns: (target_js, is_feasible) target_js has joint position to
        achieve EE target. is_feasible is if a collision joints was found.
        """

        start_state = self._mp_sim.capture_state()

        self._mp_space.set_env_state(start_state)

        start_arm_js = self._mp_sim.get_arm_pos()
        state_lims = self._mp_space.get_state_lims(True)
        lower_lims = state_lims[:, 0]
        upper_lims = state_lims[:, 1]
        found_sol = None
        for iter_i in range(timeout):
            if iter_i == 0:
                # Check if the starting state can already achieve the goal.
                cur_js = np.array(start_arm_js)
            else:
                cur_js = np.random.uniform(lower_lims, upper_lims)

            self._ik.set_arm_state(cur_js, np.zeros(cur_js.shape))
            desired_js = self._ik.calc_ik(local_ee_targ)
            self._mp_sim.set_arm_pos(desired_js)
            self._mp_sim.micro_step()

            state_valid = all(
                [self._is_state_valid_fn(desired_js) for _ in range(5)]
            )
            if state_valid:
                found_sol = np.array(desired_js)
                break

        self._mp_sim.set_arm_pos(start_arm_js)
        self._mp_sim.set_state(start_state)
        return found_sol, found_sol is not None

    def _fk(self, joints):
        self._mp_sim.set_arm_pos(joints)
        self._mp_sim.micro_step()

    def gen_target_from_ee_pos(self, ee_pos):
        inv_robo_T = self._mp_sim.get_robot_transform().inverted()
        local_ee_pos = inv_robo_T.transform_point(ee_pos)

        self.mp.setup_ee_margin(None)
        self._is_state_valid_fn = self.mp._is_state_valid

        use_js = None
        real_ee_pos = None
        for _ in range(20):
            joints, is_feasible = self._gen_goal_state(local_ee_pos)
            if not is_feasible:
                continue
            real_ee_pos = self._get_real_ee_pos(joints)
            ee_dist = np.linalg.norm(real_ee_pos - ee_pos)
            if ee_dist < self._grasp_thresh:
                use_js = joints
                break

        targ = RobotTarget(
            joints_target=use_js,
            is_guess=use_js is None,
            ee_target_pos=real_ee_pos,
        )

        self.mp.remove_ee_margin(None)
        return targ

    def _verbose_log(self, s):
        if self._is_verbose:
            print(f"GraspPlanner: {s}")

    def get_obj_goal_offset(self, obj_idx):
        obj_dat = self._mp_sim.get_obj_info(obj_idx)
        size_y = obj_dat.bb.size_y() / 2.0
        return np.array([0.0, size_y, 0.0])

    def _bounding_sphere_sample(
        self, obj_idx: int, obj_dat: ObjectGraspTarget
    ) -> RobotTarget:
        obj_pos = np.array(obj_dat.transformation.translation)

        inv_robo_T = self._mp_sim.get_robot_transform().inverted()

        # Setup extra collision checkers
        self.mp.setup_ee_margin(obj_idx)
        self._is_state_valid_fn = self.mp._is_state_valid

        # Get the candidate grasp points in global space.
        min_radius = self._grasp_thresh * 0.5

        sim = self._mp_sim._sim
        scene_obj_ids = sim.scene_obj_ids
        scene_obj_pos = sim.get_scene_pos()

        found_goal_js = None
        real_ee_pos = None

        for i in range(self._n_gen_grasps):
            self._verbose_log(f"Trying for {i}")

            # Generate a grasp 3D point
            radius = np.random.uniform(min_radius, self._grasp_thresh)
            point = np.random.randn(3)
            point[1] = np.abs(point[1])
            point = radius * (point / np.linalg.norm(point))
            point += obj_pos

            if self.knows_other_objs:
                closest_idx = np.argmin(
                    np.linalg.norm(scene_obj_pos - point, axis=-1)
                )
                if scene_obj_ids[closest_idx] != obj_idx:
                    self._verbose_log(
                        "Grasp point didn't match desired object"
                    )
                    continue

            local_point = inv_robo_T.transform_point(point)
            local_point = np.array(local_point)

            self._grasp_debug_points(obj_pos, point)

            goal_js, is_feasible = self._gen_goal_state(local_point, i)
            if not is_feasible:
                self._verbose_log("Could not find joints for grasp point")
                continue

            # Check the final end-effector position is indeed within
            # grasping position of the object.
            real_ee_pos = self._get_real_ee_pos(goal_js)

            ee_dist = np.linalg.norm(real_ee_pos - obj_pos)
            if ee_dist >= self._grasp_thresh:
                found_goal_js = goal_js
                self._verbose_log(
                    f"Actual EE wasn't in grasp range. {ee_dist} away"
                )
                continue

            if self.knows_other_objs:
                # Does the actual end-effector position grasp the object we want?
                closest_idx = np.argmin(
                    np.linalg.norm(scene_obj_pos - real_ee_pos, axis=-1)
                )
                if scene_obj_ids[closest_idx] != obj_idx:
                    self._verbose_log("Actual EE did not match desired object")
                    continue

            if self._should_render:
                sim.viz_ids["ee"] = sim.visualize_position(
                    real_ee_pos, sim.viz_ids["ee"], r=5.0
                )
                Image.fromarray(self._mp_sim.render()).save(
                    f"{self._log_dir}/grasp_plan_{i}_{ee_dist}.jpeg"
                )

            self._verbose_log(f"Found solution at {i}, breaking")
            found_goal_js = goal_js
            break

        self._clean_grasp_debug_points()
        self.mp.remove_ee_margin(obj_idx)

        return RobotTarget(
            joints_target=found_goal_js,
            obj_id_target=obj_idx,
            is_guess=found_goal_js is None,
            ee_target_pos=real_ee_pos,
        )

    def _get_real_ee_pos(self, joints):
        if joints is None:
            return None
        start_state = self._mp_sim.capture_state()
        start_js = self._mp_sim.get_arm_pos()
        self._mp_sim.set_arm_pos(joints)
        self._mp_sim.micro_step()
        real_ee_pos = self._mp_sim.get_ee_pos()
        self._mp_sim.set_arm_pos(start_js)
        self._mp_sim.set_state(start_state)
        return real_ee_pos

    def _clean_grasp_debug_points(self):
        sim = self._mp_sim._sim
        rom = sim.get_rigid_object_manager()
        if self._should_render:
            # Cleanup any debug render objects.
            if sim.viz_ids["ee"] is not None:
                rom.remove_object_by_id(sim.viz_ids["ee"])
            if sim.viz_ids["obj"] is not None:
                rom.remove_object_by_id(sim.viz_ids["obj"])
                rom.remove_object_by_id(sim.viz_ids["grasp"])

            sim.viz_ids["obj"] = None
            sim.viz_ids["grasp"] = None
            sim.viz_ids["ee"] = None

    def _grasp_debug_points(self, obj_pos, grasp_point):
        sim = self._mp_sim._sim
        if self._should_render:
            sim.viz_ids["obj"] = sim.visualize_position(
                obj_pos, sim.viz_ids["obj"], r=5.0
            )

            sim.viz_ids["grasp"] = sim.visualize_position(
                grasp_point, sim.viz_ids["grasp"], r=5.0
            )

    def gen_target_from_obj_idx(self, obj_idx):
        obj_dat = self._mp_sim.get_obj_info(obj_idx)
        return self._bounding_sphere_sample(obj_idx, obj_dat)

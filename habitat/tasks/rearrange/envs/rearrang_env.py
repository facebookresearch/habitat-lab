import copy

import attr
import magnum as mn
import numpy as np

import habitat_sim
from habitat.tasks.rearrange.envs.base_hab_env import BaseHabEnv
from habitat.tasks.rearrange.envs.utils import (
    CollDetails,
    allowed_region_to_bb,
    rearrang_collision,
)
from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_env(name="RearrangementRLEnv")
class RearrangementRLEnv(BaseHabEnv):
    """
    Defines additional logic for valid collisions and gripping.
    """

    def __init__(self, config, dataset=None):
        super().__init__(config, dataset)
        self.is_gripper_closed = False
        self._internal_sim = self._env._sim
        self.use_max_accum_force = self.tcfg.MAX_ACCUM_FORCE

    def reset(self, super_reset=True):
        self._ignore_collisions = []
        if super_reset:
            observations = super().reset()
        else:
            observations = None
        self.prev_measures = self._env.get_metrics()
        self.prev_picked = False
        self.n_succ_picks = 0
        self.coll_accum = CollDetails()
        self.prev_coll_accum = CollDetails()
        self.should_end = False
        self.accum_force = 0
        self.prev_force = None

        sim = self._env._sim

        self.allowed_region = self._env._sim.allowed_region
        task_allowed_region = self.tcfg.get("ALLOWED_REGION", [])
        if len(task_allowed_region) > 0:
            # Task allowed region overrides the scene one.
            allowed_region = task_allowed_region

            allowed_square = allowed_region[:2]
            allowed_root_art_id = allowed_region[2]
            allowed_region = allowed_region_to_bb(allowed_square)
            art_id = sim.art_name_to_id[allowed_root_art_id]
            art_T = sim._sim.get_articulated_object_root_state(art_id)

            # Transform the 2D BB
            center = allowed_region.center()
            size = allowed_region.size()
            allowed_region_3d = mn.Range3D.from_center(
                mn.Vector3(center[0], 0.0, center[1]),
                mn.Vector3(size[0] / 2, 0.5 / 2, size[1] / 2),
            )
            allowed_region_3d = habitat_sim.geo.get_transformed_bb(
                allowed_region_3d, art_T
            )
            center = allowed_region_3d.center()
            size = allowed_region_3d.size()
            self.allowed_region = mn.Range2D.from_center(
                mn.Vector2(center[0], center[2]),
                mn.Vector2(size[0] / 2, size[2] / 2),
            )

            # To visualize the allowed region.
            fence_points = [
                self.allowed_region.bottom_right,
                self.allowed_region.bottom_left,
                self.allowed_region.top_left,
                self.allowed_region.top_right,
            ]
            for i, p in enumerate(fence_points):
                full_p = [p[0], 0.5, p[1]]
                sim.viz_ids["fence_%i" % i] = sim.viz_pos(
                    full_p, sim.viz_ids["fence_%i" % i]
                )

        return observations

    def _pre_step(self):
        robo_force, _, overall_force = self._get_coll_forces()
        if self.tcfg.COUNT_OBJ_COLLISIONS:
            self.cur_force = overall_force
        else:
            self.cur_force = robo_force

        if self.prev_force is not None:
            force_diff = self.cur_force - self.prev_force
            self.add_force = 0.0
            if force_diff > 20:
                self.add_force = force_diff
                self.accum_force += self.add_force
                self.prev_force = self.cur_force
            elif force_diff < 0:
                self.prev_force = self.cur_force
        else:
            self.add_force = 0.0
            self.prev_force = self.cur_force
        self.update_coll_count()

    def _is_violating_hold_constraint(self):
        if self.tcfg.get("IGNORE_HOLD_VIOLATE", False):
            return False
        sim = self._env._sim
        # Is the object firmly in the grasp of the robot?
        hold_obj = sim.snapped_obj_id
        cur_measures = self._env.get_metrics()
        ee_pos = self._env._sim.robot.ee_transform.translation
        if hold_obj is not None:
            obj_pos = self._env._sim.get_translation(hold_obj)
            if np.linalg.norm(ee_pos - obj_pos) >= self.rlcfg.HOLD_THRESH:
                return True

        if self.tcfg.get("IGNORE_ART_HOLD_VIOLATE", False):
            return False

        art_hold_thresh = self.rlcfg.get("ART_HOLD_THRESH", 0.2)

        if sim.snapped_marker_name is not None:
            hold_pos = sim.markers[sim.snapped_marker_name]["global_pos"]
            if np.linalg.norm(ee_pos - hold_pos) >= art_hold_thresh:
                return True

        return False

    def _get_coll_reward(self):
        reward = 0
        if self.tcfg.FORCE_BASED:
            # Penalize the force that was added to the accumulated force at the
            # last time step.
            reward -= min(
                self.rlcfg.FORCE_PEN * self.add_force, self.rlcfg.MAX_FORCE_PEN
            )
        else:
            delta_coll = self._delta_coll
            reward -= self.rlcfg.ROBO_OBJ_COLL_PEN * delta_coll.robo_obj_colls

            total_colls = (
                delta_coll.obj_scene_colls + delta_coll.robo_scene_colls
            )
            if self.tcfg.COUNT_ROBO_OBJ_COLLS:
                total_colls += delta_coll.robo_obj_colls
            reward -= self.rlcfg.COLL_PEN * (min(1, total_colls))
        return reward

    def end_episode(self):
        self.should_end = True

    def step(self, action, action_args):
        obs, reward, done, info = super().step(action, action_args)
        # If we have any sort of collision at all the episode is over.
        info["ep_n_picks"] = self.n_succ_picks
        if (
            self.tcfg.MAX_COLLISIONS > 0
            and self.cur_collisions > self.tcfg.MAX_COLLISIONS
        ):
            done = True

        if self.should_end:
            done = True

        info.update(
            {"ep_" + k: v for k, v in attr.asdict(self.coll_accum).items()}
        )
        info["ep_n_collisions"] = self.cur_collisions
        info["ep_accum_force"] = self.accum_force
        self.prev_coll_accum = copy.copy(self.coll_accum)

        if self._is_violating_hold_constraint():
            reward -= self.rlcfg.CONSTRAINT_VIOLATE_PEN
            done = True
            info["ep_constraint_violate"] = 1.0
        else:
            info["ep_constraint_violate"] = 0.0

        if self.tcfg.FORCE_BASED:
            if (
                self.use_max_accum_force > 0
                and self.accum_force > self.use_max_accum_force
            ):
                reward -= self.rlcfg.FORCE_END_PEN
                done = True
                info["ep_accum_force_end"] = 1.0
            else:
                info["ep_accum_force_end"] = 0.0
        else:
            info["ep_accum_force_end"] = 0.0

        if (
            isinstance(self.allowed_region, mn.Range2D)
            and self.tcfg.END_OUT_OF_REGION
        ):
            robot_pos = self._env._sim.get_robot_transform().translation
            robot_pos = mn.Vector2(robot_pos[0], robot_pos[2])
            in_region = self.allowed_region.contains(robot_pos)
            if not in_region:
                reward -= self.rlcfg.OUT_OF_REGION_PEN
                done = True
                info["ep_out_of_region"] = 1.0
            else:
                info["ep_out_of_region"] = 0.0
        else:
            info["ep_out_of_region"] = 0.0

        return obs, reward, done, info

    @property
    def _delta_coll(self):
        d_prev = attr.asdict(self.prev_coll_accum)
        d_cur = attr.asdict(self.coll_accum)
        delta = {}
        for k in d_prev:
            delta[k] = d_cur[k] - d_prev[k]
        return CollDetails(**delta)

    def _get_coll_forces(self):
        # TODO: CANNOT GET CONTACT INFO
        return 0, 0, 0
        snapped_obj = self._env._sim.snapped_obj_id
        robo_id = self._env._sim.robot_id
        contact_points = self._env._sim._sim.get_physics_contact_points()

        def get_max_force(contact_points, check_id):
            match_contacts = [
                x
                for x in contact_points
                if x.object_id_a == check_id or x.object_id_b == check_id
            ]
            match_contacts = [
                x for x in match_contacts if x.object_id_a != x.object_id_b
            ]

            max_force = 0
            if len(match_contacts) > 0:
                max_force = max([abs(x.normal_force) for x in match_contacts])

            return max_force

        forces = [
            abs(x.normal_force)
            for x in contact_points
            if (
                x.object_id_a not in self._ignore_collisions
                and x.object_id_b not in self._ignore_collisions
            )
        ]
        max_force = max(forces) if len(forces) > 0 else 0

        max_obj_force = get_max_force(contact_points, snapped_obj)
        max_robo_force = get_max_force(contact_points, robo_id)
        return max_robo_force, max_obj_force, max_force

    def update_coll_count(self):
        colls = self._env._sim.get_collisions()
        _, coll_details = rearrang_collision(
            colls,
            self._env._sim.snapped_obj_id,
            self.tcfg.COUNT_OBJ_COLLISIONS,
        )

        self.coll_accum.obj_scene_colls += coll_details.obj_scene_colls
        self.coll_accum.robo_obj_colls += coll_details.robo_obj_colls
        self.coll_accum.robo_scene_colls += coll_details.robo_scene_colls

    @property
    def cur_collisions(self):
        ret = (
            self.coll_accum.obj_scene_colls + self.coll_accum.robo_scene_colls
        )
        if self.tcfg.COUNT_ROBO_OBJ_COLLS:
            ret += self.coll_accum.robo_obj_colls
        return ret

import copy
from typing import Any, Dict

import attr
import magnum as mn
import numpy as np
from PIL import Image

import habitat_sim
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationTask
from habitat.tasks.rearrange.envs.utils import (
    CacheHelper,
    CollDetails,
    allowed_region_to_bb,
    get_angle,
    rearrang_collision,
    reshape_obs_space,
)


class RearrangeTask(NavigationTask):
    """
    Defines additional logic for valid collisions and gripping.
    """

    def __init__(self, config, sim, dataset=None, *args, **kwargs) -> None:
        super().__init__(
            *args, config=config, sim=sim, dataset=dataset, **kwargs
        )
        self.is_gripper_closed = False
        self._sim = sim
        self.use_max_accum_force = self._config.MAX_ACCUM_FORCE
        self.n_objs = len(dataset.episodes[0].targets)

    def reset(self, episode: Episode):
        super_reset = True
        self._ignore_collisions = []
        if super_reset:
            observations = super().reset(episode)
        else:
            observations = None
        self.prev_measures = self.measurements.get_metrics()
        self.prev_picked = False
        self.n_succ_picks = 0
        self.coll_accum = CollDetails()
        self.prev_coll_accum = CollDetails()
        self.should_end = False
        self.accum_force = 0
        self.prev_force = None

        sim = self._sim

        return observations

    def _pre_step(self):
        robo_force, _, overall_force = self._get_coll_forces()
        if self._config.COUNT_OBJ_COLLISIONS:
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
        if self._config.get("IGNORE_HOLD_VIOLATE", False):
            return False
        sim = self._sim
        # Is the object firmly in the grasp of the robot?
        hold_obj = sim.snapped_obj_id
        cur_measures = self.measurements.get_metrics()
        ee_pos = self._sim.robot.ee_transform.translation
        if hold_obj is not None:
            obj_pos = self._sim.get_translation(hold_obj)
            if np.linalg.norm(ee_pos - obj_pos) >= self._config.HOLD_THRESH:
                return True

        if self._config.get("IGNORE_ART_HOLD_VIOLATE", False):
            return False

        art_hold_thresh = self._config.get("ART_HOLD_THRESH", 0.2)

        return False

    def _get_coll_reward(self):
        reward = 0
        if self._config.FORCE_BASED:
            # Penalize the force that was added to the accumulated force at the
            # last time step.
            reward -= min(
                self._config.FORCE_PEN * self.add_force,
                self._config.MAX_FORCE_PEN,
            )
        else:
            delta_coll = self._delta_coll
            reward -= (
                self._config.ROBO_OBJ_COLL_PEN * delta_coll.robo_obj_colls
            )

            total_colls = (
                delta_coll.obj_scene_colls + delta_coll.robo_scene_colls
            )
            if self._config.COUNT_ROBO_OBJ_COLLS:
                total_colls += delta_coll.robo_obj_colls
            reward -= self._config.COLL_PEN * (min(1, total_colls))
        return reward

    def step(self, action: Dict[str, Any], episode: Episode):
        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        action_name = action["action"]
        if isinstance(action_name, (int, np.integer)):
            action_name = self.get_action_name(action_name)
        assert (
            action_name in self.actions
        ), f"Can't find '{action_name}' action in {self.actions.keys()}."
        action_args = action["action_args"]
        obs = super().step(action=action, episode=episode)
        reward, done, info = 0, False, {}
        # If we have any sort of collision at all the episode is over.
        info["ep_n_picks"] = self.n_succ_picks
        if (
            self._config.MAX_COLLISIONS > 0
            and self.cur_collisions > self._config.MAX_COLLISIONS
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
            reward -= self._config.CONSTRAINT_VIOLATE_PEN
            done = True
            info["ep_constraint_violate"] = 1.0
        else:
            info["ep_constraint_violate"] = 0.0

        if self._config.FORCE_BASED:
            if (
                self.use_max_accum_force > 0
                and self.accum_force > self.use_max_accum_force
            ):
                reward -= self._config.FORCE_END_PEN
                done = True
                info["ep_accum_force_end"] = 1.0
            else:
                info["ep_accum_force_end"] = 0.0
        else:
            info["ep_accum_force_end"] = 0.0

        return obs

    @property
    def _delta_coll(self):
        d_prev = attr.asdict(self.prev_coll_accum)
        d_cur = attr.asdict(self.coll_accum)
        delta = {}
        for k in d_prev:
            delta[k] = d_cur[k] - d_prev[k]
        return CollDetails(**delta)

    def _get_coll_forces(self):
        snapped_obj = self._sim.snapped_obj_id
        robo_id = self._sim.robot_id
        contact_points = self._sim._sim.get_physics_contact_points()

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
        colls = self._sim.get_collisions()
        _, coll_details = rearrang_collision(
            colls,
            self._sim.snapped_obj_id,
            self._config.COUNT_OBJ_COLLISIONS,
        )

        self.coll_accum.obj_scene_colls += coll_details.obj_scene_colls
        self.coll_accum.robo_obj_colls += coll_details.robo_obj_colls
        self.coll_accum.robo_scene_colls += coll_details.robo_scene_colls

    @property
    def cur_collisions(self):
        ret = (
            self.coll_accum.obj_scene_colls + self.coll_accum.robo_scene_colls
        )
        if self._config.COUNT_ROBO_OBJ_COLLS:
            ret += self.coll_accum.robo_obj_colls
        return ret

    def get_n_targets(self):
        return self.n_objs

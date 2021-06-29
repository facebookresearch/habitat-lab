import copy
from typing import Any, Dict, Union

import attr
import numpy as np

from habitat.core.dataset import Episode
from habitat.tasks.nav.nav import NavigationTask
from habitat.tasks.rearrange.envs.utils import CollDetails, rearrang_collision


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
        self._done = False
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
        self._done = False
        self.add_force = 0.0

        return observations

    def _pre_step(self):
        robot_force, _, overall_force = self._get_coll_forces()
        if self._config.COUNT_OBJ_COLLISIONS:
            self.cur_force = overall_force
        else:
            self.cur_force = robot_force

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
        # Is the object firmly in the grasp of the robot?
        hold_obj = self._sim.snapped_obj_id
        ee_pos = self._sim.robot.ee_transform.translation
        if hold_obj is not None:
            obj_pos = self._sim.get_translation(hold_obj)
            if np.linalg.norm(ee_pos - obj_pos) >= self._config.HOLD_THRESH:
                return True

        if self._config.get("IGNORE_ART_HOLD_VIOLATE", False):
            return False

        return False

    def step(self, action: Dict[str, Any], episode: Episode):
        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        action_name = action["action"]
        if isinstance(action_name, (int, np.integer)):
            action_name = self.get_action_name(action_name)
        assert (
            action_name in self.actions
        ), f"Can't find '{action_name}' action in {self.actions.keys()}."

        obs = super().step(action=action, episode=episode)

        self._pre_step()

        from habitat.tasks.rearrange.rearrange_sensors import (
            RearrangePickSuccess,
        )

        if self.measurements.get_metrics()[RearrangePickSuccess.cls_uuid]:
            self._done = True

        # If we have any sort of collision at all the episode is over.
        if (
            self._config.MAX_COLLISIONS > 0
            and self.cur_collisions > self._config.MAX_COLLISIONS
        ):
            self._done = True

        if self.should_end:
            self._done = True

        self.prev_coll_accum = copy.copy(self.coll_accum)

        if self._is_violating_hold_constraint():
            self._done = True

        if (
            self._config.FORCE_BASED
            and self.use_max_accum_force > 0
            and self.accum_force > self.use_max_accum_force
        ):
            self._done = True

        return obs

    def _check_episode_is_active(
        self,
        *args: Any,
        action: Union[int, Dict[str, Any]],
        episode: Episode,
        **kwargs: Any,
    ) -> bool:
        return self._done

    @property
    def _delta_coll(self):
        d_prev = attr.asdict(self.prev_coll_accum)
        d_cur = attr.asdict(self.coll_accum)
        delta = {}
        for k in d_prev:
            delta[k] = d_cur[k] - d_prev[k]
        return CollDetails(**delta)

    def _get_coll_forces(self):
        return 0, 0, 0

        snapped_obj = self._sim.snapped_obj_id
        robot_id = self._sim.robot_id
        contact_points = self._sim._sim.get_physics_contact_points()

        def get_max_force(contact_points, check_id):
            match_contacts = [
                x
                for x in contact_points
                if check_id in [x.object_id_a, x.object_id_b]
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
        max_robot_force = get_max_force(contact_points, robot_id)
        return max_robot_force, max_obj_force, max_force

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

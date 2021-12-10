import os.path as osp
import random

import magnum as mn
import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.multi_task.dynamic_task_utils import (
    load_task_object,
)
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlDomain
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import CacheHelper, rearrange_collision

DYN_NAV_TASK_NAME = "RearrangeNavToObjTask-v0"


@registry.register_task(name=DYN_NAV_TASK_NAME)
class DynNavRLEnv(RearrangeTask):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)
        self.force_obj_idx = None
        self._prev_measure = 1.0
        self.nav_obj_name = None

        data_path = dataset.config.DATA_PATH
        mtime = osp.getmtime(data_path)
        cache_name = str(mtime) + data_path
        cache_name = cache_name.replace(".", "_")
        fname = data_path.split("/")[-1].split(".")[0]
        self.cache = CacheHelper(
            "dyn_nav_start_pos", cache_name, {}, verbose=True, rel_dir=fname
        )
        self.start_states = self.cache.load()
        self.domain = None
        self._nav_target_pos = mn.Vector3(0.0, 0.0, 0.0)
        self._nav_target_angle = 0.0

    @property
    def nav_target_pos(self):
        return self._nav_target_pos

    @property
    def nav_target_angle(self):
        return self._nav_target_angle

    def set_args(self, obj_to, orig_applied_args, **kwargs):
        self.force_obj_idx = obj_to

        self.nav_obj_name = orig_applied_args.get("obj_to", None)

    def _get_agent_pos(self):
        sim = self._env._sim
        current_pos = sim.get_robot_transform().translation
        return sim.pathfinder.snap_point(current_pos)

    def _get_cur_geo_dist(self):
        sim = self._env._sim
        distance_to_target = sim.geodesic_distance(
            self._get_agent_pos(),
            [self._nav_target_pos],
            None,
        )
        return distance_to_target

    def _determine_nav_pos(self, episode):
        cur_preds = self.domain.get_true_predicates()
        # Get all actions which can be actively applied.
        allowed_tasks = [
            action
            for action in self.domain.actions.values()
            if action.task != DYN_NAV_TASK_NAME
            and action.are_preconditions_true(cur_preds)
        ]

        use_task = random.choice(allowed_tasks)
        task_cls_name = use_task.task

        nav_point, targ_angle = self._get_nav_targ(
            task_cls_name, {"obj": 0}, episode
        )

        return nav_point, targ_angle, task_cls_name

    def _get_nav_targ(self, task_cls_name, task_args, episode):
        # Get the config for this task
        task_config = self.domain.get_task_match_for_name(
            task_cls_name
        ).task_def

        nav_point, heading_angle = get_nav_targ(
            task_cls_name,
            task_config,
            self._config.clone(),
            self._sim,
            self,
            self._dataset,
            task_args,
            episode,
        )

        return nav_point, heading_angle

    def _generate_nav_start_goal(self, episode):
        targ_pos, targ_angle, _ = self._determine_nav_pos(episode)
        orig_nav_targ_pos = self._sim.get_nav_pos(targ_pos)
        self._nav_target_pos = np.array(
            self._sim.pathfinder.snap_point(orig_nav_targ_pos)
        )

        start_pos, start_rot = get_robo_start_pos(
            self._sim, self._nav_target_pos
        )

        return (
            self._nav_target_pos,
            float(targ_angle),
            start_pos,
            float(start_rot),
        )

    def reset(self, episode: Episode):
        sim = self._sim
        super().reset(episode)

        if self.domain is None:
            self.domain = PddlDomain(
                self._config.PDDL_DOMAIN_DEF,
                self._dataset,
                self._config,
                self._sim,
            )

        episode_id = sim.ep_info["episode_id"]

        if self.force_obj_idx is not None:
            full_key = f"{episode_id}_{self.force_obj_idx}"
            if full_key in self.start_states:
                (
                    self._nav_target_pos,
                    self._nav_target_angle,
                ) = self.start_states[full_key]
            else:
                abs_true_point, task_cls_name, task_args = get_nav_from_obj_to(
                    self.nav_obj_name, self.force_obj_idx, sim
                )
                targ_pos, self._nav_target_angle = self._get_nav_targ(
                    task_cls_name, task_args, episode
                )
                orig_nav_targ_pos = sim.get_nav_pos(targ_pos)
                self._nav_target_pos = np.array(
                    sim.pathfinder.snap_point(orig_nav_targ_pos)
                )
                self._nav_target_angle = float(self._nav_target_angle)

                self.start_states[full_key] = (
                    self._nav_target_pos,
                    self._nav_target_angle,
                )
                self.cache.save(self.start_states)
            start_pos, start_rot = get_robo_start_pos(
                sim, self._nav_target_pos
            )
        else:
            if (
                episode_id in self.start_states
                and not self._config.FORCE_REGENERATE
            ):
                (
                    self._nav_target_pos,
                    self._nav_target_angle,
                    start_pos,
                    start_rot,
                ) = self.start_states[episode_id]

                sim.robot.base_pos = mn.Vector3(
                    start_pos[0],
                    sim.robot.base_pos[1],
                    start_pos[2],
                )
                sim.robot.base_rot = start_rot
            else:
                (
                    self._nav_target_pos,
                    self._nav_target_angle,
                    start_pos,
                    start_rot,
                ) = self._generate_nav_start_goal(episode)
                self.start_states[episode_id] = (
                    self._nav_target_pos,
                    self._nav_target_angle,
                    start_pos,
                    start_rot,
                )
                self.cache.save(self.start_states)

            targ_idxs, goal_pos = sim.get_targets()

        observations = super().reset(episode)

        if not sim.pathfinder.is_navigable(self._nav_target_pos):
            print("Goal is not navigable")

        if self._config.DEBUG_GOAL_POINT:
            sim.viz_ids["nav_targ_pos"] = sim.visualize_position(
                self._nav_target_pos,
                sim.viz_ids["nav_targ_pos"],
                r=10.0,
            )

        return observations


def get_nav_targ(
    task_cls_name,
    task_def_path,
    config,
    sim,
    env,
    dataset,
    task_kwargs,
    episode,
):
    orig_state = sim.capture_state(with_robot_js=True)
    load_task_object(
        task_cls_name,
        task_def_path,
        config,
        env,
        dataset,
        True,
        task_kwargs,
        episode,
    )
    robo_pos = sim.robot.base_pos
    heading_angle = sim.robot.base_rot

    sim.set_state(orig_state, set_hold=True)

    return robo_pos, heading_angle


def get_nav_from_obj_to(nav_name, obj_to, sim):
    task_args = {}

    if nav_name.startswith("TARG"):
        raise ValueError("Place task not supported yet")
    else:
        task_cls_name = "RearrangePickTask-v0"
        task_args = {"obj": obj_to}
        obj_id = sim.scene_obj_ids[obj_to]
        rom = sim.get_rigid_object_manager()
        abs_true_point = rom.get_object_by_id(
            obj_id
        ).transformation.translation

    return abs_true_point, task_cls_name, task_args


def get_robo_start_pos(sim, nav_targ_pos):
    timeout_len = 1000
    orig_state = sim.capture_state()

    # Find a valid navigable point between the start and goal.
    i = 0
    while i < timeout_len:
        start_pos = sim.pathfinder.get_random_navigable_point()
        start_rot = np.random.uniform(0, 2 * np.pi)
        sim.robot.base_pos = start_pos
        sim.robot.base_rot = start_rot
        start_island_radius = sim.pathfinder.island_radius(start_pos)
        goal_island_radius = sim.pathfinder.island_radius(nav_targ_pos)

        current_position = sim.robot.base_pos
        # This should only snap the height
        current_position = sim.pathfinder.snap_point(current_position)
        distance_to_target = sim.geodesic_distance(
            current_position, [nav_targ_pos], None
        )
        is_valid_nav = (
            start_island_radius == goal_island_radius
            and distance_to_target != np.inf
        )
        if not is_valid_nav:
            continue

        # no collision check
        for _ in range(5):
            sim.internal_step(-1)
            did_collide, details = rearrange_collision(
                sim,
                True,
                ignore_base=False,
            )
        if not did_collide:
            break
        i += 1
    if i == timeout_len - 1:
        if not is_valid_nav:
            print("Goal and start position are not navigable.")
        else:
            print("Could not get nav start without collision")
    # Reset everything except for the robot state.
    orig_state["robot_T"] = None
    sim.set_state(orig_state)
    return start_pos, start_rot

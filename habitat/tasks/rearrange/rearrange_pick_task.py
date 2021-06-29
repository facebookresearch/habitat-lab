import os.path as osp

import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationTask
from habitat.tasks.rearrange.envs.utils import (
    CacheHelper,
    get_angle,
    rearrang_collision,
)
from habitat.tasks.rearrange.rearrange_task import RearrangeTask


def merge_sim_episode_with_object_config(sim_config, episode):
    sim_config.defrost()
    sim_config.ep_info = [episode.__dict__]
    sim_config.freeze()
    return sim_config


@registry.register_task(name="RearrangePickTask-v0")
class RearrangePickTask(NavigationTask):

    """
    Embodied Question Answering Task
    Usage example:
        env = habitat.Env(config=eqa_config)

        env.reset()

        for i in range(10):
            action = sample_non_stop_action(env.action_space)
            if action["action"] != AnswerAction.name:
                env.step(action)
            metrics = env.get_metrics() # to check distance to target

        correct_answer_id = env.current_episode.question.answer_token
        env.step(
            {
                "action": AnswerAction.name,
                "action_args": {"answer_id": correct_answer_id},
            }
        )

        metrics = env.get_metrics()
    """

    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_object_config(sim_config, episode)


@registry.register_task(name="RearrangePickTask-v1")
class RearrangePickTaskV1(RearrangeTask):

    """
    Embodied Question Answering Task
    Usage example:
        env = habitat.Env(config=eqa_config)

        env.reset()

        for i in range(10):
            action = sample_non_stop_action(env.action_space)
            if action["action"] != AnswerAction.name:
                env.step(action)
            metrics = env.get_metrics() # to check distance to target

        correct_answer_id = env.current_episode.question.answer_token
        env.step(
            {
                "action": AnswerAction.name,
                "action_args": {"answer_id": correct_answer_id},
            }
        )

        metrics = env.get_metrics()
    """

    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_object_config(sim_config, episode)

    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)
        self.cache = {}

        data_path = dataset.config.DATA_PATH.format(split=dataset.config.SPLIT)

        mtime = osp.getmtime(data_path)
        cache_name = (
            str(mtime)
            + dataset.config.SPLIT
            + str(self._config.COUNT_OBJ_COLLISIONS)
        )
        cache_name += str(self._config.BASE_NOISE)
        cache_name = cache_name.replace(".", "_")

        fname = data_path.split("/")[-1].split(".")[0]

        self.cache = CacheHelper(
            "start_pos", cache_name, {}, verbose=True, rel_dir=fname
        )
        self.start_states = self.cache.load()
        self.desired_resting = np.array([0.5, 0.0, 1.0])
        self.targ_idx = None
        self.force_set_idx = None

        # TODO refactor
        # self.observation_space.spaces["obj_start_sensor"] = reshape_obs_space(
        #     self.observation_space.spaces["obj_start_sensor"], (3,)
        # )

    def _is_holding_obj(self):
        return self._sim.snapped_obj_id is not None

    def _get_targ_pos(self, sim):
        return sim.get_target_objs_start()

    def _gen_start_pos(self, sim, is_easy_init):
        target_positions = self._get_targ_pos(sim)
        sel_idx = np.random.randint(0, len(target_positions))
        if self.force_set_idx is not None:
            sel_idx = self.force_set_idx
        targ_pos = target_positions[sel_idx]

        orig_start_pos = sim.get_nav_pos(targ_pos, True)

        state = sim.capture_state()
        start_pos = orig_start_pos

        forward = np.array([1.0, 0, 0])
        dist_thresh = 0.1

        # Add a bit of noise
        timeout = 1000
        for _attempt in range(timeout):
            start_pos = orig_start_pos + np.random.normal(
                0, self._config.BASE_NOISE, size=(3,)
            )
            targ_dist = np.linalg.norm((start_pos - orig_start_pos)[[0, 2]])

            is_navigable = is_easy_init or sim._sim.pathfinder.is_navigable(
                start_pos
            )

            if targ_dist > dist_thresh or not is_navigable:
                continue

            sim.set_state(state)

            sim.set_robot_pos(start_pos[[0, 2]])

            # Face the robot towards the object.
            rel_targ = targ_pos - start_pos
            angle_to_obj = get_angle(forward[[0, 2]], rel_targ[[0, 2]])
            if np.cross(forward[[0, 2]], rel_targ[[0, 2]]) > 0:
                angle_to_obj *= -1.0
            sim.set_robot_rot(angle_to_obj)

            # Make sure the robot is not colliding with anything in this
            # position.
            for _ in range(100):
                sim.internal_step(-1)
                colls = sim.get_collisions()
                did_collide, details = rearrang_collision(
                    colls,
                    None,
                    self._config.COUNT_OBJ_COLLISIONS,
                    ignore_base=False,
                )

                if is_easy_init:
                    # Only care about collisions between the robot and scene.
                    did_collide = details.robo_scene_colls != 0

                if did_collide:
                    break

            if not did_collide:
                break

        if _attempt == timeout - 1 and (not is_easy_init):
            # print('Could not satisfy for %s! Trying with easy init' % sim.ep_info['scene_config_path'])
            start_pos, angle_to_obj, sel_idx = self._gen_start_pos(sim, True)

        sim.set_state(state)

        return start_pos, angle_to_obj, sel_idx

    def _should_prevent_grip(self, action_args):
        return (
            self._is_holding_obj()
            and action_args.get("grip_ac", None) is not None
            and action_args["grip_ac"] <= 0
        )

    def step(self, action, episode):
        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        action_name = action["action"]
        if isinstance(action_name, (int, np.integer)):
            action_name = self.get_action_name(action_name)
        assert (
            action_name in self.actions
        ), f"Can't find '{action_name}' action in {self.actions.keys()}."
        action_args = action["action_args"]

        if self._should_prevent_grip(action_args):
            # No releasing the object once it is held.
            action_args["grip_ac"] = None
        obs = super().step(action=action, episode=episode)
        obs = self._trans_obs(obs)

        return obs

    def _trans_obs(self, obs):
        if "obj_start_sensor" in obs:
            obs["obj_start_sensor"] = obs["obj_start_sensor"][self.targ_idx]
        return obs

    def set_args(self, obj, **kwargs):
        self.force_set_idx = obj

    def reset(self, episode: Episode):
        sim = self._sim

        super().reset(episode)

        self.prev_colls = 0
        episode_id = sim.ep_info["episode_id"]

        if episode_id in self.start_states and self.force_set_idx is None:
            start_pos, start_rot, sel_idx = self.start_states[episode_id]
        else:
            start_pos, start_rot, sel_idx = self._gen_start_pos(
                sim, self._config.EASY_INIT
            )
            self.start_states[episode_id] = (start_pos, start_rot, sel_idx)
            if self.force_set_idx is None:
                self.cache.save(self.start_states)

        for _ in range(5):
            sim.internal_step(-1)
            colls = sim.get_collisions()
            did_collide, _ = rearrang_collision(
                colls,
                None,
                self._config.COUNT_OBJ_COLLISIONS,
                ignore_base=False,
            )
            rot_noise = np.random.normal(0.0, self._config.BASE_ANGLE_NOISE)

            sim.set_robot_pos(start_pos[[0, 2]])
            sim.set_robot_rot(start_rot + rot_noise)
            if not did_collide:
                break

        self.targ_idx = sel_idx
        assert self.targ_idx is not None
        self.abs_targ_idx = sim.get_targets()[0][sel_idx]
        # Value < 0 will not be used. Need to do this instead of None since
        # debug renderer expects a valid number.
        self.cur_dist = -1.0
        snapped_id = self._sim.snapped_obj_id
        self.prev_picked = snapped_id is not None

        return super(RearrangePickTaskV1, self).reset(episode)

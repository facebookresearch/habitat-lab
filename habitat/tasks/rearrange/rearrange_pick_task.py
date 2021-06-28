import os
import os.path as osp

import numpy as np
from PIL import Image

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationTask
from habitat.tasks.rearrange.envs.rearrang_env import RearrangementRLEnv
from habitat.tasks.rearrange.envs.utils import (
    CacheHelper,
    get_angle,
    rearrang_collision,
    reshape_obs_space,
)
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat_baselines.common.baseline_registry import baseline_registry


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
        super().__init__(config=config, *args, **kwargs)
        # super().__init__(config, dataset)
        self.cache = {}

        data_path = config.TASK_CONFIG.DATASET.DATA_PATH.format(
            split=config.TASK_CONFIG.DATASET.SPLIT
        )

        mtime = osp.getmtime(data_path)
        cache_name = (
            str(mtime)
            + self.tcfg.DATASET.SPLIT
            + str(self.tcfg.COUNT_OBJ_COLLISIONS)
        )
        cache_name += str(self.rlcfg.BASE_NOISE)
        cache_name = cache_name.replace(".", "_")

        fname = data_path.split("/")[-1].split(".")[0]

        self.cache = CacheHelper(
            "start_pos", cache_name, {}, verbose=True, rel_dir=fname
        )
        self.start_states = self.cache.load()
        self.desired_resting = np.array([0.5, 0.0, 1.0])
        self.targ_idx = None
        self.force_set_idx = None
        self.set_force_back = None

        self.observation_space.spaces["obj_start_sensor"] = reshape_obs_space(
            self.observation_space.spaces["obj_start_sensor"], (3,)
        )

    def _is_holding_obj(self):
        return self._env._sim.snapped_obj_id is not None

    def _my_get_reward(self, observations):
        self.prev_obs = observations

        cur_measures = self._env.get_metrics()
        reward = 0

        snapped_id = self._env._sim.snapped_obj_id
        cur_picked = snapped_id is not None

        if cur_picked:
            dist_to_goal = np.linalg.norm(
                observations["ee_pos"] - self.desired_resting
            )
        else:
            dist_to_goal = cur_measures["ee_to_object_distance"][self.targ_idx]

        abs_targ_obj_idx = self._env._sim.scene_obj_ids[self.abs_targ_idx]

        did_pick = cur_picked and (not self.prev_picked)
        if did_pick:
            if snapped_id == abs_targ_obj_idx:
                self.n_succ_picks += 1
                reward += self.rlcfg.PICK_REWARD
                # If we just transitioned to the next stage our current
                # distance is stale.
                self.cur_dist = -1
            else:
                # picked the wrong object...
                reward -= self.rlcfg.WRONG_PICK_PEN
                self.should_end = True
                return reward

        if self.rlcfg.USE_DIFF:
            if self.cur_dist < 0:
                dist_diff = 0.0
            else:
                dist_diff = self.cur_dist - dist_to_goal

            # Filter out the small fluctuations
            dist_diff = round(dist_diff, 3)
            reward += self.rlcfg.DIST_REWARD * dist_diff
        else:
            reward -= self.rlcfg.DIST_REWARD * dist_to_goal
        self.cur_dist = dist_to_goal

        if not cur_picked and self.prev_picked:
            # Dropped the object...
            reward -= self.rlcfg.DROP_PEN
            self.should_end = True
            return reward

        if self._my_episode_success():
            reward += self.rlcfg.SUCC_REWARD

        reward += self._get_coll_reward()

        self.prev_picked = cur_picked

        return reward

    def _my_episode_success(self):
        # Is the agent holding the object and it's at the start?
        abs_targ_obj_idx = self._env._sim.scene_obj_ids[self.abs_targ_idx]
        cur_measures = self._env.get_metrics()
        obj_to_ee = cur_measures["ee_to_object_distance"][self.targ_idx]

        # Check that we are holding the right object and the object is actually
        # being held.
        if (
            abs_targ_obj_idx == self._env._sim.snapped_obj_id
            and obj_to_ee < self.rlcfg.HOLD_THRESH
        ):
            cur_measures = self._env.get_metrics()
            rest_dist = np.linalg.norm(
                self.prev_obs["ee_pos"] - self.desired_resting
            )
            if rest_dist < self.rlcfg.SUCC_THRESH:
                return True

        return False

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
        for attempt in range(timeout):
            start_pos = orig_start_pos + np.random.normal(
                0, self.rlcfg.BASE_NOISE, size=(3,)
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
            for i in range(100):
                sim.internal_step(-1)
                colls = sim.get_collisions()
                did_collide, details = rearrang_collision(
                    colls,
                    None,
                    self.tcfg.COUNT_OBJ_COLLISIONS,
                    ignore_base=False,
                )

                if is_easy_init:
                    # Only care about collisions between the robot and scene.
                    did_collide = details.robo_scene_colls != 0

                if did_collide:
                    break

            if not did_collide:
                break
        if attempt == timeout - 1:
            if not is_easy_init:
                # print('Could not satisfy for %s! Trying with easy init' % sim.ep_info['scene_config_path'])
                start_pos, angle_to_obj, sel_idx = self._gen_start_pos(
                    sim, True
                )
            else:
                print("collided", did_collide, details)
                print("targ dist", targ_dist)
                print(
                    "is navigable", sim._sim.pathfinder.is_navigable(start_pos)
                )
                print("was easy init", is_easy_init)
                print("Scene config", sim.ep_info["scene_config_path"])
                print("failed!")
                sim_obs = sim._sim.get_sensor_observations()
                obs = sim._sensor_suite.get_observations(sim_obs)
                save_dir = "data/inits"
                if not osp.exists(save_dir):
                    os.makedirs(save_dir)
                fname = osp.join(save_dir, "ep_%s_%i.jpeg" % ("train", 0))
                im = Image.fromarray(np.flip(obs["rgb"], 0)).save(fname)
                print(
                    "Saved image with the observation on the error episode to: ",
                    fname,
                )
                # raise ValueError('Could not generate config')

        sim.set_state(state)

        return start_pos, angle_to_obj, sel_idx

    def _should_prevent_grip(self, action_args):
        return (
            self._is_holding_obj()
            and action_args.get("grip_ac", None) is not None
            and action_args["grip_ac"] <= 0
        )

    def step(self, action, action_args):
        if self._should_prevent_grip(action_args):
            # No releasing the object once it is held.
            action_args["grip_ac"] = None
        obs, reward, done, info = super().step(action, action_args)
        info["dist_to_goal"] = self.cur_dist
        info["is_picked"] = int(self.prev_picked)
        cur_measures = self._env.get_metrics()
        info["ee_to_obj"] = cur_measures["ee_to_object_distance"][
            self.abs_targ_idx
        ]

        # only return info about the particular object that we are trying to
        # pick
        obs = self._trans_obs(obs)

        return obs, reward, done, info

    def _trans_obs(self, obs):
        if "obj_start_sensor" in obs:
            obs["obj_start_sensor"] = obs["obj_start_sensor"][self.targ_idx]
        return obs

    def set_args(self, obj, **kwargs):
        self.force_set_idx = obj
        if "force_back_pos" in kwargs:
            self.set_force_back = kwargs["force_back_pos"]

    def reset(self, episode: Episode):
        super_reset = True
        sim = self._env._sim

        if super_reset:
            sim.set_force_back(self.set_force_back)
        super().reset(episode)

        self.prev_colls = 0
        episode_id = sim.ep_info["episode_id"]

        if super_reset:
            if episode_id in self.start_states and self.force_set_idx is None:
                start_pos, start_rot, sel_idx = self.start_states[episode_id]
            else:
                start_pos, start_rot, sel_idx = self._gen_start_pos(
                    sim, self.rlcfg.EASY_INIT
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
                    self.tcfg.COUNT_OBJ_COLLISIONS,
                    ignore_base=False,
                )
                rot_noise = np.random.normal(0.0, self.rlcfg.BASE_ANGLE_NOISE)
                sim.set_robot_pos(start_pos[[0, 2]])
                sim.set_robot_rot(start_rot + rot_noise)
                if not did_collide:
                    break
        else:
            sel_idx = self.force_set_idx

        self.targ_idx = sel_idx
        assert self.targ_idx is not None
        self.abs_targ_idx = sim.get_targets()[0][sel_idx]
        # Value < 0 will not be used. Need to do this instead of None since
        # debug renderer expects a valid number.
        self.cur_dist = -1.0
        snapped_id = self._env._sim.snapped_obj_id
        self.prev_picked = snapped_id is not None

        if super_reset:
            sim.set_force_back(None)

        return self.get_task_obs()

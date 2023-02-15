#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Manually control the robot to interact with the environment. Run as
```
python examples/interative_play.py
```

To Run you need PyGame installed (to install run `pip install pygame==2.0.1`).

By default this controls with velocity control (which makes controlling the
robot hard). To use IK control instead add the `--add-ik` command line argument.

Controls:
- For velocity control
    - 1-7 to increase the motor target for the robot arm joints
    - Q-U to decrease the motor target for the robot arm joints
- For IK control
    - W,S,A,D to move side to side
    - E,Q to move up and down
- I,J,K,L to move the robot base around
- PERIOD to print the current world coordinates of the robot base.
- Z to toggle the camera to free movement mode. When in free camera mode:
    - W,S,A,D,Q,E to translate the camera
    - I,J,K,L,U,O to rotate the camera
    - B to reset the camera position
- X to change the robot that is being controlled (if there are multiple robots).

Change the task with `--cfg benchmark/rearrange/close_cab.yaml` (choose any task under the `habitat-lab/habitat/config/task/rearrange/` folder).

Change the grip type:
- Suction gripper `task.actions.arm_action.grip_controller "SuctionGraspAction"`

To record a video: `--save-obs` This will save the video to file under `data/vids/` specified by `--save-obs-fname` (by default `vid.mp4`).
"""

import argparse
import os
import os.path as osp
import time
from abc import ABC, abstractmethod

import gym.spaces as spaces
import numpy as np
import torch

import habitat
import habitat.gym.gym_wrapper as gym_wrapper
import habitat.tasks.rearrange.rearrange_task
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    GfxReplayMeasureMeasurementConfig,
    OracleNavActionConfig,
    PddlApplyActionConfig,
    ThirdRGBSensorConfig,
)
from habitat.core.logging import logger
from habitat.tasks.rearrange.actions.actions import ArmEEAction
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.config.default import get_config as get_baselines_config
from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetNet,
    PointNavResNetPolicy,
)
from habitat_baselines.rl.hrl.hierarchical_policy import (  # noqa: F401.
    HierarchicalPolicy,
)
from habitat_baselines.utils.common import get_action_space_info
from habitat_sim.utils import viz_utils as vut

try:
    import pygame
except ImportError:
    pygame = None

DEFAULT_CFG = "benchmark/rearrange/play.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
SAVE_VIDEO_DIR = "./data/vids"
SAVE_ACTIONS_DIR = "./data/interactive_play_replays"


class Controller(ABC):
    def __init__(self, agent_idx, is_multi_agent):
        self._agent_idx = agent_idx
        self._is_multi_agent = is_multi_agent

    @abstractmethod
    def act(self, obs, env):
        pass


def clean_dict(d, remove_prefix):
    ret_d = {}
    for k, v in d.spaces.items():
        if k.startswith(remove_prefix):
            new_k = k[len(remove_prefix) :]
            if isinstance(v, spaces.Dict):
                ret_d[new_k] = clean_dict(v, remove_prefix)
            else:
                ret_d[new_k] = v
        elif not k.startswith("agent"):
            ret_d[k] = v
    return spaces.Dict(ret_d)


class BaselinesController(Controller):
    def __init__(self, agent_idx, is_multi_agent, cfg_path, env):
        super().__init__(agent_idx, is_multi_agent)

        config = get_baselines_config(
            cfg_path,
            [
                "habitat_baselines/rl/policy=hl_fixed",
                "habitat_baselines/rl/policy/hierarchical_policy/defined_skills=oracle_skills",
                "habitat_baselines.num_environments=1",
            ],
        )
        policy_cls = baseline_registry.get_policy(
            config.habitat_baselines.rl.policy.name
        )
        self._env_ac = env.action_space
        env_obs = env.observation_space
        self._agent_k = f"agent_{agent_idx}_"
        if is_multi_agent:
            self._env_ac = clean_dict(self._env_ac, self._agent_k)
            env_obs = clean_dict(env_obs, self._agent_k)

        self._gym_ac_space = gym_wrapper.create_action_space(self._env_ac)
        gym_obs_space = gym_wrapper.smash_observation_space(
            env_obs, list(env_obs.keys())
        )
        self._actor_critic = policy_cls.from_config(
            config,
            gym_obs_space,
            self._gym_ac_space,
            orig_action_space=self._env_ac,
        )
        self._action_shape, _ = get_action_space_info(self._gym_ac_space)
        self._step_i = 0

    def act(self, obs, env):
        masks = torch.ones(
            (
                1,
                1,
            ),
            dtype=torch.bool,
        )
        if self._step_i == 0:
            masks = ~masks
        self._step_i += 1
        hxs = torch.ones(
            (
                1,
                1,
            ),
            dtype=torch.float32,
        )
        prev_ac = torch.ones(
            (
                1,
                self._action_shape[0],
            ),
            dtype=torch.float32,
        )
        obs = gym_wrapper.flatten_dict(obs)
        obs = TensorDict(
            {
                k[len(self._agent_k) :]: torch.tensor(v).unsqueeze(0)
                for k, v in obs.items()
                if k.startswith(self._agent_k)
            }
        )
        with torch.no_grad():
            action_data = self._actor_critic.act(obs, hxs, prev_ac, masks)
        action = gym_wrapper.continuous_vector_action_to_hab_dict(
            self._env_ac, self._gym_ac_space, action_data.env_actions[0]
        )

        def change_ac_name(k):
            if "pddl" in k:
                return k
            else:
                return self._agent_k + k

        action["action"] = [change_ac_name(k) for k in action["action"]]
        action["action_args"] = {
            change_ac_name(k): v.cpu().numpy()
            for k, v in action["action_args"].items()
        }
        return action, False, False, action_data.rnn_hidden_states


class HumanController(Controller):
    def act(self, obs, env):
        if self._is_multi_agent:
            agent_k = f"agent_{self._agent_idx}_"
        else:
            agent_k = ""
        arm_k = f"{agent_k}arm_action"
        grip_k = f"{agent_k}grip_action"
        base_k = f"{agent_k}base_vel"
        arm_name = f"{agent_k}arm_action"
        base_name = f"{agent_k}base_velocity"
        ac_spaces = env.action_space.spaces

        if arm_name in ac_spaces:
            arm_action_space = ac_spaces[arm_name][arm_k]
            arm_ctrlr = env.task.actions[arm_name].arm_ctrlr
            arm_action = np.zeros(arm_action_space.shape[0])
            grasp = 0
        else:
            arm_ctrlr = None
            arm_action = None
            grasp = None

        if base_name in ac_spaces:
            base_action_space = ac_spaces[base_name][base_k]
            base_action = np.zeros(base_action_space.shape[0])
        else:
            base_action = None

        keys = pygame.key.get_pressed()
        should_end = False
        should_reset = False

        if keys[pygame.K_ESCAPE]:
            should_end = True
        elif keys[pygame.K_m]:
            should_reset = True
        elif keys[pygame.K_n]:
            env._sim.navmesh_visualization = not env._sim.navmesh_visualization

        if base_action is not None:
            # Base control
            if keys[pygame.K_j]:
                # Left
                base_action = [0, 1]
            elif keys[pygame.K_l]:
                # Right
                base_action = [0, -1]
            elif keys[pygame.K_k]:
                # Back
                base_action = [-1, 0]
            elif keys[pygame.K_i]:
                # Forward
                base_action = [1, 0]

        if isinstance(arm_ctrlr, ArmEEAction):
            EE_FACTOR = 0.5
            # End effector control
            if keys[pygame.K_d]:
                arm_action[1] -= EE_FACTOR
            elif keys[pygame.K_a]:
                arm_action[1] += EE_FACTOR
            elif keys[pygame.K_w]:
                arm_action[0] += EE_FACTOR
            elif keys[pygame.K_s]:
                arm_action[0] -= EE_FACTOR
            elif keys[pygame.K_q]:
                arm_action[2] += EE_FACTOR
            elif keys[pygame.K_e]:
                arm_action[2] -= EE_FACTOR
        else:
            # Velocity control. A different key for each joint
            if keys[pygame.K_q]:
                arm_action[0] = 1.0
            elif keys[pygame.K_1]:
                arm_action[0] = -1.0

            elif keys[pygame.K_w]:
                arm_action[1] = 1.0
            elif keys[pygame.K_2]:
                arm_action[1] = -1.0

            elif keys[pygame.K_e]:
                arm_action[2] = 1.0
            elif keys[pygame.K_3]:
                arm_action[2] = -1.0

            elif keys[pygame.K_r]:
                arm_action[3] = 1.0
            elif keys[pygame.K_4]:
                arm_action[3] = -1.0

            elif keys[pygame.K_t]:
                arm_action[4] = 1.0
            elif keys[pygame.K_5]:
                arm_action[4] = -1.0

            elif keys[pygame.K_y]:
                arm_action[5] = 1.0
            elif keys[pygame.K_6]:
                arm_action[5] = -1.0

            elif keys[pygame.K_u]:
                arm_action[6] = 1.0
            elif keys[pygame.K_7]:
                arm_action[6] = -1.0

        if keys[pygame.K_p]:
            logger.info("[play.py]: Unsnapping")
            # Unsnap
            grasp = -1
        elif keys[pygame.K_o]:
            # Snap
            logger.info("[play.py]: Snapping")
            grasp = 1

        if keys[pygame.K_PERIOD]:
            # Print the current position of the robot, useful for debugging.
            pos = [
                float("%.3f" % x) for x in env._sim.robot.sim_obj.translation
            ]
            rot = env._sim.robot.sim_obj.rotation
            ee_pos = env._sim.robot.ee_transform.translation
            logger.info(
                f"Robot state: pos = {pos}, rotation = {rot}, ee_pos = {ee_pos}"
            )
        elif keys[pygame.K_COMMA]:
            # Print the current arm state of the robot, useful for debugging.
            joint_state = [
                float("%.3f" % x) for x in env._sim.robot.arm_joint_pos
            ]
            logger.info(f"Robot arm joint state: {joint_state}")

        action_names = []
        action_args = {}
        if base_action is not None:
            action_names.append(base_name)
            action_args.update(
                {
                    base_k: base_action,
                }
            )
        if arm_action is not None:
            action_names.append(arm_name)
            action_args.update(
                {
                    arm_k: arm_action,
                    grip_k: grasp,
                }
            )
        if len(action_names) == 0:
            raise ValueError("No active actions for human controller.")

        return (
            {"action": action_names, "action_args": action_args},
            should_reset,
            should_end,
            {},
        )


def play_env(env, args, config):
    render_steps_limit = None
    if args.no_render:
        render_steps_limit = DEFAULT_RENDER_STEPS_LIMIT

    obs = env.reset()

    if not args.no_render:
        draw_obs = observations_to_image(obs, {})
        pygame.init()
        screen = pygame.display.set_mode(
            [draw_obs.shape[1], draw_obs.shape[0]]
        )

    update_idx = 0
    target_fps = 60.0
    prev_time = time.time()
    all_obs = []
    total_reward = 0

    gfx_measure = env.task.measurements.measures.get(
        GfxReplayMeasure.cls_uuid, None
    )
    is_multi_agent = len(env._sim.robots_mgr) > 1

    controllers = []
    n_robots = len(env._sim.robots_mgr)
    all_hxs = [None for _ in range(n_robots)]
    active_controllers = [0, 1]
    controllers = [
        HumanController(0, is_multi_agent),
        BaselinesController(
            1,
            is_multi_agent,
            "habitat-baselines/habitat_baselines/config/rearrange/rl_hierarchical.yaml",
            env,
        ),
    ]

    while True:
        if render_steps_limit is not None and update_idx > render_steps_limit:
            break

        keys = pygame.key.get_pressed()

        if not args.no_render and keys[pygame.K_x]:
            active_controllers[0] = (active_controllers[0] + 1) % n_robots
            logger.info(
                f"Controlled agent changed. Controlling agent {active_controllers[0]}."
            )

        end_play = False
        reset_ep = False
        if args.no_render:
            action = {"action": "empty", "action_args": {}}
        else:
            all_names = []
            all_args = {}
            for i in active_controllers:
                (
                    ctrl_action,
                    ctrl_reset_ep,
                    ctrl_end_play,
                    all_hxs[i],
                ) = controllers[i].act(obs, env)
                end_play = end_play or ctrl_end_play
                reset_ep = reset_ep or ctrl_reset_ep
                all_names.extend(ctrl_action["action"])
                all_args.update(ctrl_action["action_args"])
            action = {"action": tuple(all_names), "action_args": all_args}

        obs = env.step(action)

        if not args.no_render and keys[pygame.K_c]:
            pddl_action = env.task.actions["PDDL_APPLY_ACTION"]
            logger.info("Actions:")
            actions = pddl_action._action_ordering
            for i, action in enumerate(actions):
                logger.info(f"{i}: {action}")
            entities = pddl_action._entities_list
            logger.info("Entities")
            for i, entity in enumerate(entities):
                logger.info(f"{i}: {entity}")
            action_sel = input("Enter Action Selection: ")
            entity_sel = input("Enter Entity Selection: ")
            action_sel = int(action_sel)
            entity_sel = [int(x) + 1 for x in entity_sel.split(",")]
            ac = np.zeros(pddl_action.action_space["pddl_action"].shape[0])
            ac_start = pddl_action.get_pddl_action_start(action_sel)
            ac[ac_start : ac_start + len(entity_sel)] = entity_sel

            env.step(
                {
                    "action": "PDDL_APPLY_ACTION",
                    "action_args": {"pddl_action": ac},
                }
            )

        if not args.no_render and keys[pygame.K_g]:
            pred_list = env.task.sensor_suite.sensors[
                "all_predicates"
            ]._predicates_list
            pred_values = obs["all_predicates"]
            logger.info("\nPredicate Truth Values:")
            for i, (pred, pred_value) in enumerate(
                zip(pred_list, pred_values)
            ):
                logger.info(f"{i}: {pred.compact_str} = {pred_value}")

        if reset_ep:
            total_reward = 0
            # Clear the saved keyframes.
            if gfx_measure is not None:
                gfx_measure.get_metric(force_get=True)
            env.reset()
        if end_play:
            break

        update_idx += 1

        info = env.get_metrics()
        reward_key = [k for k in info if "reward" in k]
        if len(reward_key) > 0:
            reward = info[reward_key[0]]
        else:
            reward = 0.0

        total_reward += reward
        info["Total Reward"] = total_reward

        use_ob = observations_to_image(obs, info)
        if not args.skip_render_text:
            use_ob = overlay_frame(use_ob, info)

        draw_ob = use_ob[:]

        if not args.no_render:
            draw_ob = np.transpose(draw_ob, (1, 0, 2))
            draw_obuse_ob = pygame.surfarray.make_surface(draw_ob)
            screen.blit(draw_obuse_ob, (0, 0))
            pygame.display.update()
        if args.save_obs:
            all_obs.append(draw_ob)  # type: ignore[assignment]

        if not args.no_render:
            pygame.event.pump()
        if env.episode_over:
            total_reward = 0
            env.reset()

        curr_time = time.time()
        diff = curr_time - prev_time
        delay = max(1.0 / target_fps - diff, 0)
        time.sleep(delay)
        prev_time = curr_time

    if args.save_obs:
        all_obs = np.array(all_obs)  # type: ignore[assignment]
        all_obs = np.transpose(all_obs, (0, 2, 1, 3))  # type: ignore[assignment]
        os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)
        vut.make_video(
            np.expand_dims(all_obs, 1),
            0,
            "color",
            osp.join(SAVE_VIDEO_DIR, args.save_obs_fname),
        )
    if gfx_measure is not None and args.gfx:
        gfx_str = gfx_measure.get_metric(force_get=True)
        write_gfx_replay(
            gfx_str, config.habitat.task, env.current_episode.episode_id
        )

    if not args.no_render:
        pygame.quit()


def has_pygame():
    return pygame is not None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--save-obs", action="store_true", default=False)
    parser.add_argument("--save-obs-fname", type=str, default="play.mp4")
    parser.add_argument("--play-cam-res", type=int, default=512)
    parser.add_argument(
        "--skip-render-text", action="store_true", default=False
    )
    parser.add_argument(
        "--same-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--skip-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )
    parser.add_argument(
        "--disable-inverse-kinematics",
        action="store_true",
        help="If specified, does not add the inverse kinematics end-effector control.",
    )
    parser.add_argument(
        "--gfx",
        action="store_true",
        default=False,
        help="Save a GFX replay file.",
    )
    parser.add_argument("--load-actions", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    if not has_pygame() and not args.no_render:
        raise ImportError(
            "Need to install PyGame (run `pip install pygame==2.0.1`)"
        )

    config = habitat.get_config(args.cfg, args.opts)
    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task
        task_config.actions["pddl_apply_action"] = PddlApplyActionConfig()
        task_config.actions[
            "agent_1_oracle_nav_action"
        ] = OracleNavActionConfig(agent_index=1)

        if not args.same_task:
            sim_config.debug_render = True
            agent_config = get_agent_config(sim_config=sim_config)
            agent_config.sim_sensors.update(
                {
                    "third_rgb_sensor": ThirdRGBSensorConfig(
                        height=args.play_cam_res, width=args.play_cam_res
                    )
                }
            )
            if "composite_success" in task_config.measurements:
                task_config.measurements.composite_success.must_call_stop = (
                    False
                )
            if "rearrange_nav_to_obj_success" in task_config.measurements:
                task_config.measurements.rearrange_nav_to_obj_success.must_call_stop = (
                    False
                )
            if "force_terminate" in task_config.measurements:
                task_config.measurements.force_terminate.max_accum_force = -1.0
                task_config.measurements.force_terminate.max_instant_force = (
                    -1.0
                )

        if args.gfx:
            sim_config.habitat_sim_v0.enable_gfx_replay_save = True
            task_config.measurements.update(
                {"gfx_replay_measure": GfxReplayMeasureMeasurementConfig()}
            )

        if args.never_end:
            env_config.max_episode_steps = 0

        if not args.disable_inverse_kinematics:
            if "arm_action" not in task_config.actions:
                raise ValueError(
                    "Action space does not have any arm control so cannot add inverse kinematics. Specify the `--disable-inverse-kinematics` option"
                )
            sim_config.agents.main_agent.ik_arm_urdf = (
                "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
            )
            task_config.actions.arm_action.arm_controller = "ArmEEAction"

    with habitat.Env(config=config) as env:
        play_env(env, args, config)

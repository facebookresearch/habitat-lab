#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any

import gym.spaces as spaces
import magnum as mn
import numpy as np
import torch

import habitat.gym.gym_wrapper as gym_wrapper
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.gui.gui_input import GuiInput
from habitat.tasks.rearrange.actions.actions import ArmEEAction
from habitat.utils.common import flatten_dict
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.config.default import get_config as get_baselines_config
from habitat_baselines.utils.common import get_action_space_info


class Controller(ABC):
    def __init__(self, agent_idx, is_multi_agent):
        self._agent_idx = agent_idx
        self._is_multi_agent = is_multi_agent

    @abstractmethod
    def act(self, obs, env):
        pass

    def on_environment_reset(self):
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
                # "habitat_baselines/rl/policy=hl_fixed",
                # "habitat_baselines/rl/policy/hierarchical_policy/defined_skills=oracle_skills",
                "habitat_baselines/rl/policy/hierarchical_policy/defined_skills@habitat_baselines.rl.policy.main_agent.hierarchical_policy.defined_skills=oracle_skills",
                "habitat_baselines.num_environments=1",
            ],
        )
        policy_cls = baseline_registry.get_policy(
            config.habitat_baselines.rl.policy.main_agent.name
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
        obs = flatten_dict(obs)
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

        # temp do random base actions
        action["action_args"]["base_vel"] = torch.rand_like(
            action["action_args"]["base_vel"]
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


class GuiRobotController(Controller):
    def __init__(self, agent_idx, is_multi_agent, gui_input):
        super().__init__(agent_idx, is_multi_agent)
        self._gui_input = gui_input

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

        base_action: Any = None
        if base_name in ac_spaces:
            base_action_space = ac_spaces[base_name][base_k]
            base_action = np.zeros(base_action_space.shape[0])
        else:
            base_action = None

        KeyNS = GuiInput.KeyNS
        gui_input = self._gui_input

        should_end = False
        should_reset = False

        if gui_input.get_key_down(KeyNS.ESC):
            should_end = True
        elif gui_input.get_key_down(KeyNS.M):
            should_reset = True
        elif gui_input.get_key_down(KeyNS.N):
            env._sim.navmesh_visualization = not env._sim.navmesh_visualization

        if base_action is not None:
            # Base control
            base_action = [0, 0]
            if gui_input.get_key(KeyNS.J):
                # Left
                base_action[1] += 1
            if gui_input.get_key(KeyNS.L):
                # Right
                base_action[1] -= 1
            if gui_input.get_key(KeyNS.K):
                # Back
                base_action[0] -= 1
            if gui_input.get_key(KeyNS.I):
                # Forward
                base_action[0] += 1

        if isinstance(arm_ctrlr, ArmEEAction):
            EE_FACTOR = 0.5
            # End effector control
            if gui_input.get_key_down(KeyNS.D):
                arm_action[1] -= EE_FACTOR
            elif gui_input.get_key_down(KeyNS.A):
                arm_action[1] += EE_FACTOR
            elif gui_input.get_key_down(KeyNS.W):
                arm_action[0] += EE_FACTOR
            elif gui_input.get_key_down(KeyNS.S):
                arm_action[0] -= EE_FACTOR
            elif gui_input.get_key_down(KeyNS.Q):
                arm_action[2] += EE_FACTOR
            elif gui_input.get_key_down(KeyNS.E):
                arm_action[2] -= EE_FACTOR
        else:
            # Velocity control. A different key for each joint
            if gui_input.get_key_down(KeyNS.Q):
                arm_action[0] = 1.0
            elif gui_input.get_key_down(KeyNS.ONE):
                arm_action[0] = -1.0

            elif gui_input.get_key_down(KeyNS.W):
                arm_action[1] = 1.0
            elif gui_input.get_key_down(KeyNS.TWO):
                arm_action[1] = -1.0

            elif gui_input.get_key_down(KeyNS.E):
                arm_action[2] = 1.0
            elif gui_input.get_key_down(KeyNS.THREE):
                arm_action[2] = -1.0

            elif gui_input.get_key_down(KeyNS.R):
                arm_action[3] = 1.0
            elif gui_input.get_key_down(KeyNS.FOUR):
                arm_action[3] = -1.0

            elif gui_input.get_key_down(KeyNS.T):
                arm_action[4] = 1.0
            elif gui_input.get_key_down(KeyNS.FIVE):
                arm_action[4] = -1.0

            elif gui_input.get_key_down(KeyNS.Y):
                arm_action[5] = 1.0
            elif gui_input.get_key_down(KeyNS.SIX):
                arm_action[5] = -1.0

            elif gui_input.get_key_down(KeyNS.U):
                arm_action[6] = 1.0
            elif gui_input.get_key_down(KeyNS.SEVEN):
                arm_action[6] = -1.0

        if gui_input.get_key_down(KeyNS.P):
            # logger.info("[play.py]: Unsnapping")
            # Unsnap
            grasp = -1
        elif gui_input.get_key_down(KeyNS.O):
            # Snap
            # logger.info("[play.py]: Snapping")
            grasp = 1

        # reference code
        # if gui_input.get_key_down(KeyNS.PERIOD):
        #     # Print the current position of the robot, useful for debugging.
        #     pos = [
        #         float("%.3f" % x) for x in env._sim.robot.sim_obj.translation
        #     ]
        #     rot = env._sim.robot.sim_obj.rotation
        #     ee_pos = env._sim.robot.ee_transform.translation
        #     logger.info(
        #         f"Robot state: pos = {pos}, rotation = {rot}, ee_pos = {ee_pos}"
        #     )
        # elif gui_input.get_key_down(KeyNS.COMMA):
        #     # Print the current arm state of the robot, useful for debugging.
        #     # joint_state = [
        #     #     float("%.3f" % x) for x in env._sim.robot.arm_joint_pos
        #     # ]

        #     # logger.info(f"Robot arm joint state: {joint_state}")

        action_names = []
        action_args: Any = {}
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


class GuiHumanoidController(Controller):
    def __init__(
        self, agent_idx, is_multi_agent, gui_input, env, walk_pose_path
    ):
        self.agent_idx = agent_idx
        super().__init__(self.agent_idx, is_multi_agent)
        self._humanoid_controller = HumanoidRearrangeController(walk_pose_path)
        self._env = env
        self._gui_input = gui_input
        self._hint_walk_dir = None
        self._hint_grasp_obj_idx = None
        self._hint_drop_pos = None

    def get_articulated_agent(self):
        return self._env._sim.agents_mgr[self.agent_idx].articulated_agent

    def on_environment_reset(self):
        super().on_environment_reset()
        base_pos = self.get_articulated_agent().base_pos
        self._humanoid_controller.reset(base_pos)
        self._hint_walk_dir = None
        self._hint_grasp_obj_idx = None
        self._hint_drop_pos = None

    def get_random_joint_action(self):
        # Add random noise to human arms but keep global transform
        (
            joint_trans,
            root_trans,
        ) = self.get_articulated_agent().get_joint_transform()
        # Divide joint_trans by 4 since joint_trans has flattened quaternions
        # and the dimension of each quaternion is 4
        num_joints = len(joint_trans) // 4
        root_trans = np.array(root_trans)
        index_arms_start = 10
        joint_trans_quat = [
            mn.Quaternion(
                mn.Vector3(joint_trans[(4 * index) : (4 * index + 3)]),
                joint_trans[4 * index + 3],
            )
            for index in range(num_joints)
        ]
        rotated_joints_quat = []
        for index, joint_quat in enumerate(joint_trans_quat):
            random_vec = np.random.rand(3)
            # We allow for maximum 10 angles per step
            random_angle = np.random.rand() * 10
            rotation_quat = mn.Quaternion.rotation(
                mn.Rad(random_angle), mn.Vector3(random_vec).normalized()
            )
            if index > index_arms_start:
                joint_quat *= rotation_quat
            rotated_joints_quat.append(joint_quat)
        joint_trans = np.concatenate(
            [
                np.array(list(quat.vector) + [quat.scalar])
                for quat in rotated_joints_quat
            ]
        )
        humanoidjoint_action = np.concatenate(
            [joint_trans.reshape(-1), root_trans.transpose().reshape(-1)]
        )
        return humanoidjoint_action

    def set_act_hints(self, walk_dir, grasp_obj_idx, do_drop):
        self._hint_walk_dir = walk_dir
        self._hint_grasp_obj_idx = grasp_obj_idx
        self._hint_drop_pos = do_drop

    def _get_grasp_mgr(self):
        agents_mgr = self._env._sim.agents_mgr
        grasp_mgr = agents_mgr._all_agent_data[self.agent_idx].grasp_mgr
        return grasp_mgr

    @property
    def is_grasped(self):
        return self._get_grasp_mgr().is_grasped

    def _update_grasp(self, grasp_object_id, drop_pos):
        if grasp_object_id is not None:
            assert not self.is_grasped
            self._get_grasp_mgr().snap_to_obj(grasp_object_id)

        elif drop_pos:
            assert self.is_grasped
            grasp_object_id = self._get_grasp_mgr().snap_idx
            self._get_grasp_mgr().desnap()

            # teleport to requested drop_pos
            sim = self._env.task._sim
            rigid_obj = sim.get_rigid_object_manager().get_object_by_id(
                grasp_object_id
            )
            rigid_obj.translation = drop_pos

    def act(self, obs, env):
        self._update_grasp(self._hint_grasp_obj_idx, self._hint_drop_pos)
        self._hint_grasp_obj_idx = None
        self._hint_drop_pos = None

        if self._is_multi_agent:
            agent_k = f"agent_{self._agent_idx}_"
        else:
            agent_k = ""
        humanoidjoint_name = f"{agent_k}humanoidjoint_action"
        ac_spaces = env.action_space.spaces

        do_humanoidjoint_action = humanoidjoint_name in ac_spaces

        KeyNS = GuiInput.KeyNS
        gui_input = self._gui_input

        should_end = False
        should_reset = False

        if gui_input.get_key_down(KeyNS.ESC):
            should_end = True
        elif gui_input.get_key_down(KeyNS.M):
            should_reset = True
        elif gui_input.get_key_down(KeyNS.N):
            # todo: move outside this controller
            env._sim.navmesh_visualization = not env._sim.navmesh_visualization

        if do_humanoidjoint_action:
            humancontroller_base_user_input = [0, 0]
            # temp keyboard controls to test humanoid controller
            if gui_input.get_key(KeyNS.I):
                # move in world-space x+ direction ("east")
                humancontroller_base_user_input[0] += 1
            if gui_input.get_key(KeyNS.K):
                # move in world-space x- direction ("west")
                humancontroller_base_user_input[0] -= 1

            if self._hint_walk_dir:
                humancontroller_base_user_input[0] += self._hint_walk_dir.x
                humancontroller_base_user_input[1] += self._hint_walk_dir.z

        action_names = []
        action_args = {}
        if do_humanoidjoint_action:
            if True:
                relative_pos = mn.Vector3(
                    humancontroller_base_user_input[0],
                    0,
                    humancontroller_base_user_input[1],
                )
                # pose, root_trans = self._humanoid_controller.get_walk_pose(
                #     relative_pos, distance_multiplier=1.0
                # )
                # humanoidjoint_action = self._humanoid_controller.vectorize_pose(
                #     pose, root_trans
                # )
                # Use the controller
                self._humanoid_controller.calculate_walk_pose(relative_pos)
                humanoidjoint_action = self._humanoid_controller.get_pose()
            else:
                pass
                # reference code
                # humanoidjoint_action = self.get_random_joint_action()
            action_names.append(humanoidjoint_name)
            action_args.update(
                {
                    "agent_0_human_joints_trans": humanoidjoint_action,
                }
            )

        return (
            {"action": action_names, "action_args": action_args},
            should_reset,
            should_end,
            {},
        )


class ControllerHelper:
    def __init__(self, env, args, gui_input):
        self.n_robots = len(env._sim.agents_mgr)
        is_multi_agent = self.n_robots > 1

        self._env = env

        gui_controller: Controller = None
        if args.humanoid_user_agent:
            gui_controller = GuiHumanoidController(
                0, is_multi_agent, gui_input, env, args.walk_pose_path
            )
        else:
            gui_controller = GuiRobotController(0, is_multi_agent, gui_input)

        self.controllers = []
        self.n_robots = self.n_robots
        self.all_hxs = [None for _ in range(self.n_robots)]
        self.active_controllers = [0, 1]
        self.controllers = [
            gui_controller,
            BaselinesController(
                1,
                is_multi_agent,
                "habitat-baselines/habitat_baselines/config/rearrange/rl_hierarchical.yaml",
                env,
            ),
        ]

    def get_gui_humanoid_controller(self):
        if isinstance(self.controllers[0], GuiHumanoidController):
            return self.controllers[0]
        return None

    def update(self, obs):
        all_names = []
        all_args = {}
        end_play = False
        reset_ep = False
        for i in self.active_controllers:
            (
                ctrl_action,
                ctrl_reset_ep,
                ctrl_end_play,
                self.all_hxs[i],
            ) = self.controllers[i].act(obs, self._env)
            end_play = end_play or ctrl_end_play
            reset_ep = reset_ep or ctrl_reset_ep
            all_names.extend(ctrl_action["action"])
            all_args.update(ctrl_action["action_args"])
        action = {"action": tuple(all_names), "action_args": all_args}
        return action, end_play, reset_ep

    def on_environment_reset(self):
        for i in self.active_controllers:
            self.controllers[i].on_environment_reset()

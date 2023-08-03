#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import gym.spaces as spaces
import magnum as mn
import numpy as np
import torch

import habitat.gym.gym_wrapper as gym_wrapper
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.gui.gui_input import GuiInput
from habitat.tasks.rearrange.actions.actions import ArmEEAction
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.rl.multi_agent.multi_agent_access_mgr import (
    MultiAgentAccessMgr,
)
from habitat_baselines.rl.multi_agent.utils import (
    update_dict_with_agent_prefix,
)
from habitat_baselines.rl.ppo.single_agent_access_mgr import (
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    get_action_space_info,
    is_continuous_action_space,
)


class Controller(ABC):
    def __init__(self, is_multi_agent):
        self._is_multi_agent = is_multi_agent

    @abstractmethod
    def act(self, obs, env):
        pass

    def on_environment_reset(self):
        pass


class GuiController(Controller):
    def __init__(self, agent_idx, is_multi_agent, gui_input):
        super().__init__(is_multi_agent)
        self._agent_idx = agent_idx
        self._gui_input = gui_input


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
    def __init__(
        self,
        is_multi_agent,
        config,
        gym_habitat_env,
        sample_random_baseline_base_vel=False,
    ):
        super().__init__(is_multi_agent)
        self._sample_random_baseline_base_vel = sample_random_baseline_base_vel

        self._config = config

        self._gym_habitat_env = gym_habitat_env
        self._habitat_env = gym_habitat_env.unwrapped.habitat_env
        self._num_envs = 1

        self.device = (
            torch.device("cuda", config.habitat_baselines.torch_gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # create env spec
        self._env_spec = self._create_env_spec()

        # create observations transforms
        self._obs_transforms = self._get_active_obs_transforms()

        # apply observations transforms
        self._env_spec.observation_space = apply_obs_transforms_obs_space(
            self._env_spec.observation_space, self._obs_transforms
        )

        # create agent
        self._agent = self._create_agent()
        if (
            self._agent.actor_critic.should_load_agent_state
            and self._config.habitat_baselines.eval.should_load_ckpt
        ):
            self._load_agent_checkpoint()

        self._action_shape, self._discrete_actions = get_action_space_info(
            self._agent.policy_action_space
        )

        self._agent.eval()

        hidden_state_lens = self._agent.hidden_state_shape_lens
        action_space_lens = self._agent.policy_action_space_shape_lens

        self._space_lengths = {}
        n_agents = len(self._config.habitat.simulator.agents)
        if n_agents > 1:
            self._space_lengths = {
                "index_len_recurrent_hidden_states": hidden_state_lens,
                "index_len_prev_actions": action_space_lens,
            }

    @abstractmethod
    def _create_env_spec(self):
        pass

    @abstractmethod
    def _get_active_obs_transforms(self):
        pass

    @abstractmethod
    def _create_agent(self):
        pass

    @abstractmethod
    def _load_agent_state_dict(self, checkpoint):
        pass

    def _load_agent_checkpoint(self):
        checkpoint = torch.load(
            self._config.habitat_baselines.eval_ckpt_path_dir,
            map_location="cpu",
        )
        self._load_agent_state_dict(checkpoint)

    def _batch_and_apply_transforms(self, obs):
        batch = batch_obs(obs, device=self.device)
        batch = apply_obs_transforms_batch(batch, self._obs_transforms)

        return batch

    def on_environment_reset(self):
        self._test_recurrent_hidden_states = torch.zeros(
            (
                self._num_envs,
                *self._agent.hidden_state_shape,
            ),
            device=self.device,
        )

        self._prev_actions = torch.zeros(
            self._num_envs,
            *self._action_shape,
            device=self.device,
            dtype=torch.long if self._discrete_actions else torch.float,
        )

        self._not_done_masks = torch.zeros(
            (
                self._num_envs,
                *self._agent.masks_shape,
            ),
            device=self.device,
            dtype=torch.bool,
        )

    def act(self, obs, env):
        batch = self._batch_and_apply_transforms([obs])

        with torch.no_grad():
            action_data = self._agent.actor_critic.act(
                batch,
                self._test_recurrent_hidden_states,
                self._prev_actions,
                self._not_done_masks,
                deterministic=False,
                **self._space_lengths,
            )
            if action_data.should_inserts is None:
                self._test_recurrent_hidden_states = (
                    action_data.rnn_hidden_states
                )
                self._prev_actions.copy_(action_data.actions)  # type: ignore
            else:
                self._agent.update_hidden_state(
                    self._test_recurrent_hidden_states,
                    self._prev_actions,
                    action_data,
                )

        if is_continuous_action_space(self._env_spec.action_space):
            # Clipping actions to the specified limits
            step_data = [
                np.clip(
                    a.numpy(),
                    self._env_spec.action_space.low,
                    self._env_spec.action_space.high,
                )
                for a in action_data.env_actions.cpu()
            ]
        else:
            step_data = [a.item() for a in action_data.env_actions.cpu()]

        action = gym_wrapper.continuous_vector_action_to_hab_dict(
            self._env_spec.orig_action_space,
            self._env_spec.action_space,
            step_data[0],
        )

        # temp: do random base actions
        if self._sample_random_baseline_base_vel:
            action["action_args"]["base_vel"] = torch.rand_like(
                action["action_args"]["base_vel"]
            )

        return action


class SingleAgentBaselinesController(BaselinesController):
    def __init__(
        self,
        agent_idx,
        is_multi_agent,
        config,
        gym_habitat_env,
        sample_random_baseline_base_vel=False,
    ):
        self._agent_idx = agent_idx
        self._agent_name = config.habitat.simulator.agents_order[
            self._agent_idx
        ]
        if is_multi_agent:
            self._agent_k = f"agent_{self._agent_idx}_"
        else:
            self._agent_k = ""

        super().__init__(
            is_multi_agent,
            config,
            gym_habitat_env,
            sample_random_baseline_base_vel,
        )

    def _create_env_spec(self):
        # udjust the observation and action space to be agent specific (remove other agents)
        original_action_space = clean_dict(
            self._gym_habitat_env.original_action_space, self._agent_k
        )
        observation_space = clean_dict(
            self._gym_habitat_env.observation_space, self._agent_k
        )
        action_space = gym_wrapper.create_action_space(original_action_space)

        env_spec = EnvironmentSpec(
            observation_space=observation_space,
            action_space=action_space,
            orig_action_space=original_action_space,
        )

        return env_spec

    def _get_active_obs_transforms(self):
        return get_active_obs_transforms(self._config, self._agent_name)

    def _create_agent(self):
        agent = SingleAgentAccessMgr(
            agent_name=self._agent_name,
            config=self._config,
            env_spec=self._env_spec,
            num_envs=self._num_envs,
            is_distrib=False,
            device=self.device,
            percent_done_fn=lambda: 0,
        )

        return agent

    def _load_agent_state_dict(self, checkpoint):
        self._agent.load_state_dict(checkpoint[self._agent_idx])

    def _batch_and_apply_transforms(self, obs):
        batch = super()._batch_and_apply_transforms(obs)
        batch = update_dict_with_agent_prefix(batch, self._agent_idx)

        return batch

    def act(self, obs, env):
        action = super().act(obs, env)

        def change_ac_name(k):
            return self._agent_k + k

        action["action"] = [change_ac_name(k) for k in action["action"]]
        action["action_args"] = {
            change_ac_name(k): v for k, v in action["action_args"].items()
        }

        return action


class MultiAgentBaselinesController(BaselinesController):
    def _create_env_spec(self):
        observation_space = self._gym_habitat_env.observation_space
        action_space = self._gym_habitat_env.action_space
        original_action_space = self._gym_habitat_env.original_action_space

        env_spec = EnvironmentSpec(
            observation_space=observation_space,
            action_space=action_space,
            orig_action_space=original_action_space,
        )

        return env_spec

    def _get_active_obs_transforms(self):
        return get_active_obs_transforms(self._config)

    def _create_agent(self):
        agent = MultiAgentAccessMgr(
            config=self._config,
            env_spec=self._env_spec,
            num_envs=self._num_envs,
            is_distrib=False,
            device=self.device,
            percent_done_fn=lambda: 0,
        )

        return agent

    def _load_agent_state_dict(self, checkpoint):
        self._agent.load_state_dict(checkpoint)


class GuiRobotController(GuiController):
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

        return {"action": action_names, "action_args": action_args}


class GuiHumanoidController(GuiController):
    def __init__(
        self,
        agent_idx,
        is_multi_agent,
        gui_input,
        env,
        walk_pose_path,
        recorder,
    ):
        super().__init__(agent_idx, is_multi_agent, gui_input)
        self._humanoid_controller = HumanoidRearrangeController(walk_pose_path)
        self._env = env
        self._hint_walk_dir = None
        self._hint_grasp_obj_idx = None
        self._hint_drop_pos = None
        self._cam_yaw = 0
        self._saved_object_rotation = None
        self._recorder = recorder

    def get_articulated_agent(self):
        return self._env._sim.agents_mgr[self._agent_idx].articulated_agent

    def on_environment_reset(self):
        super().on_environment_reset()
        base_trans = self.get_articulated_agent().base_transformation
        self._humanoid_controller.reset(base_trans)
        self._hint_walk_dir = None
        self._hint_grasp_obj_idx = None
        self._hint_drop_pos = None
        self._cam_yaw = 0
        assert not self.is_grasped

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

    def set_act_hints(self, walk_dir, grasp_obj_idx, do_drop, cam_yaw=None):
        self._hint_walk_dir = walk_dir
        self._hint_grasp_obj_idx = grasp_obj_idx
        self._hint_drop_pos = do_drop
        self._cam_yaw = cam_yaw

    def _get_grasp_mgr(self):
        agents_mgr = self._env._sim.agents_mgr
        grasp_mgr = agents_mgr._all_agent_data[self._agent_idx].grasp_mgr
        return grasp_mgr

    @property
    def is_grasped(self):
        return self._get_grasp_mgr().is_grasped

    def _update_grasp(self, grasp_object_id, drop_pos):
        if grasp_object_id is not None:
            assert not self.is_grasped

            sim = self._env.task._sim
            rigid_obj = sim.get_rigid_object_manager().get_object_by_id(
                grasp_object_id
            )
            self._saved_object_rotation = rigid_obj.rotation

            self._get_grasp_mgr().snap_to_obj(grasp_object_id)

            self._recorder.record("grasp_object_id", grasp_object_id)

        elif drop_pos is not None:
            assert self.is_grasped
            grasp_object_id = self._get_grasp_mgr().snap_idx
            self._get_grasp_mgr().desnap()

            # teleport to requested drop_pos
            sim = self._env.task._sim
            rigid_obj = sim.get_rigid_object_manager().get_object_by_id(
                grasp_object_id
            )
            rigid_obj.translation = drop_pos
            rigid_obj.rotation = self._saved_object_rotation
            self._saved_object_rotation = None

            self._recorder.record("drop_pos", drop_pos)

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

        if do_humanoidjoint_action:
            humancontroller_base_user_input = np.zeros(3)
            # temp keyboard controls to test humanoid controller
            if gui_input.get_key(KeyNS.W):
                # walk forward in the camera yaw direction
                humancontroller_base_user_input[0] += 1
            if gui_input.get_key(KeyNS.S):
                # walk forward in the opposite to camera yaw direction
                humancontroller_base_user_input[0] -= 1

            if self._hint_walk_dir:
                humancontroller_base_user_input[0] += self._hint_walk_dir.x
                humancontroller_base_user_input[2] += self._hint_walk_dir.z

                self._recorder.record("hint_walk_dir", self._hint_walk_dir)

            else:
                self._recorder.record("cam_yaw", self._cam_yaw)
                self._recorder.record(
                    "walk_forward_back", humancontroller_base_user_input[0]
                )

                rot_y_rad = -self._cam_yaw + np.pi
                rot_y_matrix = np.array(
                    [
                        [np.cos(rot_y_rad), 0, np.sin(rot_y_rad)],
                        [0, 1, 0],
                        [-np.sin(rot_y_rad), 0, np.cos(rot_y_rad)],
                    ]
                )
                humancontroller_base_user_input = (
                    rot_y_matrix @ humancontroller_base_user_input
                )

            self._recorder.record(
                "base_user_input", humancontroller_base_user_input
            )

        action_names = []
        action_args = {}
        if do_humanoidjoint_action:
            if True:
                relative_pos = mn.Vector3(humancontroller_base_user_input)

                base_offset = self.get_articulated_agent().params.base_offset
                # base_offset is basically the offset from the humanoid's root (often
                # located near its pelvis) to the humanoid's feet (where it should
                # snap to the navmesh), for example (0, -0.9, 0).
                prev_query_pos = (
                    self._humanoid_controller.obj_transform_base.translation
                    + base_offset
                )

                self._humanoid_controller.calculate_walk_pose(relative_pos)

                # calculate_walk_pose has updated obj_transform_base.translation with
                # desired motion, but this should be filtered (restricted to navmesh).
                target_query_pos = (
                    self._humanoid_controller.obj_transform_base.translation
                    + base_offset
                )
                filtered_query_pos = self._env._sim.step_filter(
                    prev_query_pos, target_query_pos
                )
                # fixup is the difference between the movement allowed by step_filter
                # and the requested base movement.
                fixup = filtered_query_pos - target_query_pos
                self._humanoid_controller.obj_transform_base.translation += (
                    fixup
                )

                humanoidjoint_action = self._humanoid_controller.get_pose()
            else:
                pass
                # reference code
                # humanoidjoint_action = self.get_random_joint_action()
            action_names.append(humanoidjoint_name)
            action_args.update(
                {
                    f"{agent_k}human_joints_trans": humanoidjoint_action,
                }
            )

        return {"action": action_names, "action_args": action_args}


class ControllerHelper:
    def __init__(self, gym_habitat_env, config, args, gui_input, recorder):
        self._gym_habitat_env = gym_habitat_env
        self._env = gym_habitat_env.unwrapped.habitat_env
        self._gui_controlled_agent_index = args.gui_controlled_agent_index

        self.n_agents = len(self._env._sim.agents_mgr)
        self.n_user_controlled_agents = (
            0 if self._gui_controlled_agent_index is None else 1
        )
        self.n_policy_controlled_agents = (
            self.n_agents - self.n_user_controlled_agents
        )
        is_multi_agent = self.n_agents > 1

        self.controllers: List[Controller] = []
        if self.n_agents == self.n_policy_controlled_agents:
            self.controllers.append(
                MultiAgentBaselinesController(
                    is_multi_agent,
                    config,
                    self._gym_habitat_env,
                    sample_random_baseline_base_vel=args.sample_random_baseline_base_vel,
                )
            )
        else:
            self.controllers.append(
                SingleAgentBaselinesController(
                    0,
                    is_multi_agent,
                    config,
                    self._gym_habitat_env,
                    sample_random_baseline_base_vel=args.sample_random_baseline_base_vel,
                )
            )

        if self._gui_controlled_agent_index is not None:
            agent_name = self._env.sim.habitat_config.agents_order[
                self._gui_controlled_agent_index
            ]
            articulated_agent_type = self._env.sim.habitat_config.agents[
                agent_name
            ].articulated_agent_type

            gui_agent_controller: Controller
            if articulated_agent_type == "KinematicHumanoid":
                gui_agent_controller = GuiHumanoidController(
                    agent_idx=self._gui_controlled_agent_index,
                    is_multi_agent=is_multi_agent,
                    gui_input=gui_input,
                    env=self._env,
                    walk_pose_path=args.walk_pose_path,
                    recorder=recorder.get_nested_recorder("gui_humanoid"),
                )
            else:
                gui_agent_controller = GuiRobotController(
                    agent_idx=self._gui_controlled_agent_index,
                    is_multi_agent=is_multi_agent,
                    gui_input=gui_input,
                )
            self.controllers.insert(
                self._gui_controlled_agent_index, gui_agent_controller
            )

        self.active_controllers = list(
            range(len(self.controllers))
        )  # assuming all controllers are active

    def get_gui_agent_controller(self) -> Optional[Controller]:
        if self._gui_controlled_agent_index is None:
            return None

        return self.controllers[self._gui_controlled_agent_index]

    def get_gui_controlled_agent_index(self) -> Optional[int]:
        return self._gui_controlled_agent_index

    def update(self, obs):
        all_names = []
        all_args = {}
        for i in self.active_controllers:
            ctrl_action = self.controllers[i].act(obs, self._env)
            all_names.extend(ctrl_action["action"])
            all_args.update(ctrl_action["action_args"])
        action = {"action": tuple(all_names), "action_args": all_args}

        return action

    def on_environment_reset(self):
        for i in self.active_controllers:
            self.controllers[i].on_environment_reset()

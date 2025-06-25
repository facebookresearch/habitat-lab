# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import numpy.typing as npt

from habitat.mochi.mochi_visualizer import MochiVisualizer

from mochi_gym.envs.allegro_in_hand_env import AllegroInHandEnv
from mochi_gym.envs.gym_base_env_cfg import GymBaseEnvCfg
from mochi_gym.envs.mochi_env import MochiEnv
from mochi_gym.envs.mochi_env_cfg import RenderMode

def get_action(time: float, action_shape: int) -> npt.NDArray[float]:
    period_length = 1.0
    index = int(time / period_length) % action_shape
    value = 0.5 * np.pi * np.sin(time / period_length * 2 * np.pi)
    action = np.zeros(action_shape)
    action[index] = value
    return action

class MochiWrapper:
    def __init__(self, hab_sim):

        self._env = AllegroInHandEnv(GymBaseEnvCfg(render_mode=RenderMode.NONE, steps_per_episode=-1))

        self._env.reset()

        self._action_sampler=get_action

        # self._object_handles = {}
        # names = self._env._mochi.get_actors_names()

        # # remove non-rigid objects; eventually, our visualizer render_map will handle this
        # names.remove("Agent")
        # names.remove("StaticPlane")

        # for name in names:
        #     handle = self._env._mochi.get_actor_handle(name)
        #     self._object_handles[name] = handle
    
        self._mochi_visualizer = MochiVisualizer(hab_sim, self._env._mochi)

        self._mochi_visualizer.add_render_map("./data_vc/mochi/test_scene.render_map.json")

        pass

    # @property
    # def service(self):
    #     return self._service

    # # must call this before rendering in habitat-sim!
    # def pre_render(self):
    #     if self._service.usd_visualizer:
    #         self._service.usd_visualizer.flush_to_hab_sim()

    def get_allegro_joint_positions(self):

        pose = self._env._mochi.get_agent_pose()
        assert len(pose) == 22
        # skip over base position and rot vector (first 6 items)
        return pose[6:]        


    # def get_object_poses(self):

    #     def rotvec_to_quat_wxyz(rotvec):
    #         theta = np.linalg.norm(rotvec)
    #         if theta < 1e-8:
    #             # No rotation, return identity quaternion
    #             return np.array([1.0, 0.0, 0.0, 0.0])  # [x, y, z, w]
            
    #         axis = rotvec / theta
    #         half_theta = theta / 2.0
    #         sin_half_theta = np.sin(half_theta)
    #         cos_half_theta = np.cos(half_theta)
            
    #         q_xyz = axis * sin_half_theta
    #         q_w = cos_half_theta
    #         return np.concatenate([[q_w], q_xyz])

    #     num_objects = len(self._object_handles)

    #     positions = np.zeros((num_objects, 3), dtype=np.float32)  # perf todo: use np.empty
    #     rotations_wxyz = np.zeros((num_objects, 4), dtype=np.float32)  # perf todo: use np.empty
    #     for (i, name) in enumerate(self._object_handles):
    #         handle = self._object_handles[name]
    #         pose = self._env._mochi.get_object_com_transform(handle)
    #         positions[i] = pose[0]
    #         rotations_wxyz[i] = rotvec_to_quat_wxyz(pose[1])

    #     # todo: coordinate frame conversion?
    #     # todo: use true object origin instead of CoM
    #     return positions, rotations_wxyz


    def step(self, num_steps=1):

        env = self._env

        action = self._action_sampler(
            env._step_count * env._sim_dt, env.action_space.shape[0]
        )
        # Send the computed action to the env.
        _, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset()

    def pre_render(self):

        self._mochi_visualizer.flush_to_hab_sim()


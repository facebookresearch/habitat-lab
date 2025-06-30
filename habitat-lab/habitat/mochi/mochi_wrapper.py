# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np
import numpy.typing as npt
from mochi_gym.envs.allegro_in_hand_env import (
    AllegroInHandEnv,
    AllegroInHandEnvCfg,
)
from mochi_gym.envs.gym_base_env_cfg import GymBaseEnvCfg
from mochi_gym.envs.mochi_env import MochiEnv
from mochi_gym.envs.mochi_env_cfg import RenderMode

from habitat.mochi.mochi_utils import habitat_to_mochi_position, quat_to_rotvec
from habitat.mochi.mochi_visualizer import MochiVisualizer

# from habitat.mochi.mochi_debug_drawer import MochiDebugDrawer


class MochiWrapper:
    def __init__(self, hab_sim, do_render=False):
        environment = AllegroInHandEnv(
            AllegroInHandEnvCfg(
                steps_per_episode=-1,
                render_mode=RenderMode.HUMAN if do_render else RenderMode.NONE,
                simulation_frequency=100,
                control_frequency=100,
                do_rgb=False,
                worker_index=0,
                # require for our position controller to work
                force_identity_agent_pose=True,
            )
        )

        self._env = environment

        self._env.reset()

        # self._object_handles = {}
        # names = self._env._mochi.get_actors_names()

        # # remove non-rigid objects; eventually, our visualizer render_map will handle this
        # names.remove("Agent")
        # names.remove("StaticPlane")

        # for name in names:
        #     handle = self._env._mochi.get_actor_handle(name)
        #     self._object_handles[name] = handle

        self._mochi_visualizer = MochiVisualizer(hab_sim, self._env._mochi)

        self._mochi_visualizer.add_render_map(
            "./data_vc/mochi/test_scene.render_map.json"
        )

        # render_actors =
        # self._debug_drawer = MochiDebugDrawer(


    def _update_metahand_and_get_action(self, dummy_metahand):
        target_pose = self._env._previous_target_pose
        target_pose[0:3] = list(
            habitat_to_mochi_position(dummy_metahand.target_base_position)
        )
        target_pose[3:6] = quat_to_rotvec(dummy_metahand.target_base_rotation)

        # note different finger convention

        # source
        # 0 pointer twist
        # 1 thumb rotate
        # 2 ring twist
        # 3 pinky twist
        # 4 pointer base
        # 5 thumb twist
        # 6 ring base
        # 7 pinky base
        # 8 pointer mid
        # 9 thumb base
        # 10 ring mid
        # 11 pinky mid
        # 12 pointer tip
        # 13 thumb tip
        # 14 ring tip
        # 15 pinky tip

        # dest
        # 0..4 finger twist and thumb rotate
        # 4..8 finger first joint bend and thumb twist
        # 8..12 finger second joint bend and thumb first joint bend
        # 12..16 last joint bend

        # 0,4,8,12 -> thumb rotate, thumb twist, then bend
        # 1,5,9,13 -> pinky twist and joint bends
        # 2,6,10,14 -> ring twist and joint bends
        # 3,7,11,15 -> pointer twist and joint bends

        remap = {
            0: 3,
            1: 0,
            2: 2,
            3: 1,
            4: 7,
            5: 4,
            6: 6,
            7: 5,
            8: 11,
            9: 8,
            10: 10,
            11: 9,
            12: 15,
            13: 12,
            14: 14,
            15: 13,
        }

        start = 6
        for src in remap:
            dest = remap[src]
            target_pose[start + dest] = dummy_metahand._target_joint_positions[
                src
            ]

        # target_pose[6:22] = [0.0] * 16  # temp

        # target_pose[6:22] = dummy_metahand._target_joint_positions

        action = [0.0] * 16
        return action

    def step(self, dummy_metahand):
        env = self._env

        action = self._update_metahand_and_get_action(dummy_metahand)

        # Send the computed action to the env.
        _, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset()

    def pre_render(self):
        self._mochi_visualizer.flush_to_hab_sim()

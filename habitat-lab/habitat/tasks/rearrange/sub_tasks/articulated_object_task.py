#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Any, Dict, Tuple

import magnum as mn
import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.marker_info import MarkerInfo
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import rearrange_collision, rearrange_logger
from habitat.tasks.utils import get_angle


class SetArticulatedObjectTask(RearrangeTask):
    """
    Base class for all tasks involving manipulating articulated objects.
    """

    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)
        self._use_marker: str = None
        self._prev_awake = True
        self._force_use_marker = None

    @property
    def use_marker_name(self) -> str:
        """
        The name of the target marker the agent interacts with.
        """
        return self._use_marker

    def get_use_marker(self) -> MarkerInfo:
        """
        The marker the agent should interact with.
        """
        return self._sim.get_marker(self._use_marker)

    def set_args(self, marker, obj, **kwargs):
        if marker.startswith("MARKER_"):
            marker = marker[len("MARKER_") :]
        self._force_use_marker = marker
        # The object in the container we are trying to reach and using as the
        # position of the container.
        self._targ_idx = obj

    @property
    def success_js_state(self) -> float:
        """
        The success state of the articulated object desired joint.
        """
        return self._config.success_state

    @abstractmethod
    def _gen_start_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_look_pos(self) -> np.ndarray:
        """
        The point defining where the robot should face at the start of the
        episode.
        """

    @abstractmethod
    def _get_spawn_region(self) -> mn.Range2D:
        """
        The region on the ground the robot can be placed.
        """

    def _sample_robot_start(self, T) -> Tuple[float, np.ndarray]:
        """
        Returns the start face direction and the starting position of the robot.
        """
        spawn_region = self._get_spawn_region()

        if self._config.spawn_region_scale == 0.0:
            # No randomness in the base position spawn
            start_pos = spawn_region.center()
        else:
            spawn_region = mn.Range2D.from_center(
                spawn_region.center(),
                self._config.spawn_region_scale * spawn_region.size() / 2,
            )

            start_pos = np.random.uniform(spawn_region.min, spawn_region.max)

        start_pos = np.array([start_pos[0], 0.0, start_pos[1]])
        targ_pos = np.array(self._get_look_pos())

        # Transform to global coordinates
        start_pos = np.array(T.transform_point(mn.Vector3(*start_pos)))
        start_pos = np.array([start_pos[0], 0, start_pos[2]])
        start_pos = self._sim.safe_snap_point(start_pos)

        targ_pos = np.array(T.transform_point(mn.Vector3(*targ_pos)))

        # Spawn the robot facing the look pos
        forward = np.array([1.0, 0, 0])
        rel_targ = targ_pos - start_pos
        angle_to_obj = get_angle(forward[[0, 2]], rel_targ[[0, 2]])
        if np.cross(forward[[0, 2]], rel_targ[[0, 2]]) > 0:
            angle_to_obj *= -1.0

        return angle_to_obj, start_pos

    def step(self, action: Dict[str, Any], episode: Episode):
        return super().step(action, episode)

    @property
    def _is_there_spawn_noise(self):
        return (
            self._config.base_angle_noise != 0.0
            or self._config.spawn_region_scale != 0.0
        )

    def reset(self, episode: Episode):
        super().reset(episode, fetch_observations=False)
        if self._force_use_marker is not None:
            self._use_marker = self._force_use_marker

        marker = self.get_use_marker()
        if self._config.use_marker_t:
            T = marker.get_current_transform()
        else:
            ao = marker.ao_parent
            T = ao.transformation

        jms = marker.ao_parent.get_joint_motor_settings(marker.joint_idx)

        if self._config.joint_max_impulse > 0:
            jms.velocity_target = 0.0
            jms.max_impulse = self._config.joint_max_impulse
        marker.ao_parent.update_joint_motor(marker.joint_idx, jms)

        num_timeout = 100
        self._disable_art_sleep()
        for _ in range(num_timeout):
            self._set_link_state(self._gen_start_state())

            angle_to_obj, base_pos = self._sample_robot_start(T)

            noise = np.random.normal(0.0, self._config.base_angle_noise)
            self._sim.articulated_agent.base_rot = angle_to_obj + noise
            self._sim.articulated_agent.base_pos = base_pos

            articulated_agent_T = (
                self._sim.articulated_agent.base_transformation
            )
            rel_targ_pos = articulated_agent_T.inverted().transform_point(
                marker.current_transform.translation
            )
            if not self._is_there_spawn_noise:
                rearrange_logger.debug(
                    "No spawn noise, returning first found position"
                )
                break

            eps = 1e-2
            upper_bound = (
                self._sim.articulated_agent.params.ee_constraint[0, :, 1] + eps
            )
            is_within_bounds = (rel_targ_pos < upper_bound).all()
            if not is_within_bounds:
                continue

            did_collide = False
            for _ in range(self._config.settle_steps):
                self._sim.internal_step(-1)
                did_collide, details = rearrange_collision(
                    self._sim,
                    self._config.count_obj_collisions,
                    ignore_base=False,
                )
                if did_collide:
                    break
            if not did_collide:
                break

        # Step so the updated art position evaluates
        self._sim.internal_step(-1)
        self._reset_art_sleep()

        self.prev_dist_to_push = -1

        self.prev_snapped_marker_name = None
        self._sim.maybe_update_articulated_agent()
        return self._get_observations(episode)

    def _disable_art_sleep(self):
        """
        Disables the sleeping state of the articulated object. Use when setting
        the articulated object joint states.
        """
        ao = self.get_use_marker().ao_parent
        self._prev_awake = ao.awake
        ao.awake = True

    def _reset_art_sleep(self) -> None:
        """
        Resets the sleeping state of the target articulated object.
        """
        ao = self.get_use_marker().ao_parent
        ao.awake = self._prev_awake

    def _set_link_state(self, art_pos: np.ndarray) -> None:
        """
        Set the joint state of all the joints on the target articulated object.
        """
        ao = self.get_use_marker().ao_parent
        ao.joint_positions = art_pos


@registry.register_task(name="RearrangeOpenDrawerTask-v0")
class RearrangeOpenDrawerTaskV1(SetArticulatedObjectTask):
    def _get_spawn_region(self):
        return mn.Range2D([0.80, -0.35], [0.95, 0.35])

    def _get_look_pos(self):
        return [0.0, 0.0, 0.0]

    def _gen_start_state(self):
        drawers = np.zeros((8,))
        return drawers

    def reset(self, episode: Episode):
        self._use_marker = "cab_push_point_5"
        return super().reset(episode)


@registry.register_task(name="RearrangeOpenFridgeTask-v0")
class RearrangeOpenFridgeTaskV1(SetArticulatedObjectTask):
    def _get_spawn_region(self):
        return mn.Range2D([0.833, -0.6], [1.25, 0.6])

    def _get_look_pos(self):
        return [0.0, 0.0, 0.0]

    def _gen_start_state(self):
        return np.zeros((2,))

    def reset(self, episode: Episode):
        self._use_marker = "fridge_push_point"
        return super().reset(episode)


@registry.register_task(name="RearrangeCloseDrawerTask-v0")
class RearrangeCloseDrawerTaskV1(SetArticulatedObjectTask):
    def _get_spawn_region(self):
        back_x = 0.8
        # How far back the robot should be from the drawer.
        return mn.Range2D([back_x, -0.35], [back_x + 0.05, 0.35])

    def _get_look_pos(self):
        return [0.0, 0.0, 0.0]

    def _gen_start_state(self):
        targ_link = self.get_use_marker().joint_idx

        drawers = np.zeros((8,))
        drawers[targ_link] = np.random.uniform(0.4, 0.5)
        return drawers

    def reset(self, episode: Episode):
        self._use_marker = "cab_push_point_5"
        return super().reset(episode)


@registry.register_task(name="RearrangeCloseFridgeTask-v0")
class RearrangeCloseFridgeTaskV1(SetArticulatedObjectTask):
    def _get_spawn_region(self):
        return mn.Range2D([0.833, -0.6], [1.25, 0.6])

    def _get_look_pos(self):
        return [0.0, 0.0, 0.0]

    def _gen_start_state(self):
        return np.array([0, np.random.uniform(np.pi / 4, 2 * np.pi / 3)])

    def reset(self, episode: Episode):
        self._use_marker = "fridge_push_point"
        return super().reset(episode)

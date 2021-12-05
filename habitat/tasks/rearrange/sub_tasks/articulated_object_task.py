from abc import abstractmethod
from typing import Any, Dict

import magnum as mn
import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import rearrange_collision
from habitat.tasks.utils import get_angle


class SetArticulatedObjectTask(RearrangeTask):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)
        self._use_marker: str = None
        self._prev_awake = True
        self._force_use_marker = None

    @property
    def use_marker_name(self):
        return self._use_marker

    def get_use_marker(self):
        return self._sim.get_marker(self._use_marker)

    def set_args(self, marker, **kwargs):
        self._force_use_marker = marker

    @property
    def success_js_state(self):
        return self._config.SUCCESS_STATE

    @abstractmethod
    def _gen_start_state(self):
        pass

    @abstractmethod
    def _get_look_pos(self):
        """
        The point defining where the robot should face at the start of the
        episode.
        """

    @abstractmethod
    def _get_spawn_region(self) -> mn.Range2D:
        pass

    def _sample_robot_start(self, T):
        spawn_region = self._get_spawn_region()
        spawn_region = mn.Range2D.from_center(
            spawn_region.center(),
            self._config.SPAWN_REGION_SCALE * spawn_region.size() / 2,
        )

        start_pos = np.random.uniform(spawn_region.min, spawn_region.max)

        start_pos = np.array([start_pos[0], 0.0, start_pos[1]])
        targ_pos = np.array(self._get_look_pos())

        # Transform to global coordinates
        start_pos = np.array(T.transform_point(mn.Vector3(*start_pos)))
        start_pos = np.array([start_pos[0], 0, start_pos[2]])

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

    def reset(self, episode: Episode):
        super().reset(episode)
        if self._force_use_marker is not None:
            self._use_marker = self._force_use_marker

        marker = self.get_use_marker()
        if self._config.USE_MARKER_T:
            T = marker.get_current_transform()
        else:
            ao = marker.ao_parent
            T = ao.transformation

        jms = marker.ao_parent.get_joint_motor_settings(marker.joint_idx)

        if self._config.JOINT_MAX_IMPULSE > 0:
            jms.velocity_target = 0.0
            jms.max_impulse = self._config.JOINT_MAX_IMPULSE
        marker.ao_parent.update_joint_motor(marker.joint_idx, jms)

        num_timeout = 100
        num_pos_timeout = 100
        self._disable_art_sleep()
        for _ in range(num_timeout):
            self._set_link_state(self._gen_start_state())

            for _ in range(num_pos_timeout):
                angle_to_obj, start_pos = self._sample_robot_start(T)
                if self._sim.pathfinder.is_navigable(start_pos):
                    break

            noise = np.random.normal(0.0, self._config.BASE_ANGLE_NOISE)
            self._sim.robot.base_rot = angle_to_obj + noise
            base_pos = mn.Vector3(
                start_pos[0],
                self._sim.robot.base_pos[1],
                start_pos[2],
            )
            self._sim.robot.base_pos = base_pos

            did_collide = False
            for _ in range(self._config.SETTLE_STEPS):
                self._sim.internal_step(-1)
                did_collide, details = rearrange_collision(
                    self._sim,
                    self._config.COUNT_OBJ_COLLISIONS,
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
        return super().reset(episode)

    def _disable_art_sleep(self):
        ao = self.get_use_marker().ao_parent
        self._prev_awake = ao.awake
        ao.awake = True

    def _reset_art_sleep(self):
        ao = self.get_use_marker().ao_parent
        ao.awake = self._prev_awake

    def _set_link_state(self, art_pos):
        ao = self.get_use_marker().ao_parent
        ao.joint_positions = art_pos

    def _get_art_pos(self):
        return self.get_use_marker().ao_parent.transformation.translation


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

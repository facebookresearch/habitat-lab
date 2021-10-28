from abc import abstractmethod

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

    @property
    def _is_start_global(self):
        return False

    @abstractmethod
    def _sample_pos(self) -> np.ndarray:
        """
        Returns a 2D vector for the robot start position
        """

    def _sample_robot_start(self, T):
        start_pos = self._sample_pos()
        start_pos = np.array([start_pos[0], 0.0, start_pos[1]])
        targ_pos = np.array(self._get_look_pos())

        if not self._is_start_global:
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
                start_pos[1],
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
    def _sample_pos(self):
        # How far back the robot should be from the drawer.
        return np.random.uniform([0.80, -0.35], [0.95, 0.35])

    def _get_look_pos(self):
        return [0.0, 0.0, 0.0]

    def _gen_start_state(self):
        drawers = np.zeros((8,))
        # num_open = np.random.randint(0,6)
        # poss_idxs = list(range(7))
        # del poss_idxs[poss_idxs.index(self.targ_art_idx)]
        # random.shuffle(poss_idxs)
        # open_idxs = poss_idxs[:num_open]

        # drawers[open_idxs] = np.random.uniform(0.0, 0.3, size=(7,))[open_idxs]
        # drawers[self.targ_art_idx] = 0.0
        return drawers

    def reset(self, episode: Episode):
        self._use_marker = "cab_push_point_5"
        return super().reset(episode)


@registry.register_task(name="RearrangeCloseDrawerTask-v0")
class RearrangeCloseDrawerTaskV1(SetArticulatedObjectTask):
    def _sample_pos(self):
        back_x = 0.8
        # How far back the robot should be from the drawer.
        return np.random.uniform([back_x, -0.35], [back_x + 0.05, 0.35])

    def _get_look_pos(self):
        return [0.0, 0.0, 0.0]

    def _gen_start_state(self):
        targ_link = self.get_use_marker().joint_idx

        drawers = np.zeros((8,))
        # num_open = np.random.randint(0, 7)
        # poss_idxs = list(range(8))
        # del poss_idxs[poss_idxs.index(targ_link)]
        # random.shuffle(poss_idxs)
        # open_idxs = poss_idxs[:num_open]

        # drawers[open_idxs] = np.random.uniform(0.0, 0.1, size=(7,))[open_idxs]
        drawers[targ_link] = np.random.uniform(0.4, 0.5)
        return drawers

    def reset(self, episode: Episode):
        self._use_marker = "cab_push_point_5"
        return super().reset(episode)

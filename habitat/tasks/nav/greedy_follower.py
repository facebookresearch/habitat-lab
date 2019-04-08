from typing import Union

import habitat_sim
import numpy as np
import quaternion

from habitat.sims.habitat_simulator import HabitatSim
from habitat.sims.habitat_simulator import SIM_NAME_TO_ACTION, SimulatorActions


def v_to_q(v: np.array):
    return np.quaternion(v[3], *v[0:3])


def action_to_one_hot(action: int) -> np.array:
    one_hot = np.zeros(len(SIM_NAME_TO_ACTION), dtype=np.float32)
    one_hot[action] = 1
    return one_hot


class GreedyActionPathFollower:
    def __init__(
        self, sim: HabitatSim, goal_radius: float, return_one_hot: bool = True
    ):
        assert (
            getattr(sim, "geodesic_distance", None) is not None
        ), "{} must have a method called geodesic_distance".format(
            type(sim).__name__
        )

        self._sim = sim
        self._max_delta = self._sim.config.FORWARD_STEP_SIZE - 1e6
        self._goal_radius = goal_radius
        self._step_size = self._sim.config.FORWARD_STEP_SIZE

        self._mode = (
            "spath"
            if getattr(sim, "straight_spath_points", None) is not None
            else "grad"
        )
        self._return_one_hot = return_one_hot

    def _get_return_value(
        self, action: SimulatorActions
    ) -> Union[SimulatorActions, np.array]:
        if self._return_one_hot:
            return action_to_one_hot(SIM_NAME_TO_ACTION[action.value])
        else:
            return action

    def get_next_action(
        self, goal_pos: np.array
    ) -> Union[SimulatorActions, np.array]:
        if (
            np.linalg.norm(goal_pos - self._sim.get_agent_state().position)
            <= self._goal_radius
        ):
            return self._get_return_value(SimulatorActions.STOP)

        max_grad_dir = self._est_max_grad_dir(goal_pos)
        if max_grad_dir is None:
            return self._get_return_value(SimulatorActions.FORWARD)
        return self.step_along_grad(max_grad_dir, goal_pos)

    def step_along_grad(
        self, grad_dir: np.quaternion, goal_pos: np.array
    ) -> Union[SimulatorActions, np.array]:
        current_state = self._sim.get_agent_state()
        alpha = self._angle_between_quats(
            grad_dir, v_to_q(current_state.rotation)
        )
        if alpha <= np.deg2rad(self._sim.config.TURN_ANGLE) + 1e-3:
            return self._get_return_value(SimulatorActions.FORWARD)
        else:
            sim_action = SIM_NAME_TO_ACTION[SimulatorActions.LEFT.value]
            self._sim.step(sim_action)
            best_turn = (
                SimulatorActions.LEFT
                if (
                    self._angle_between_quats(
                        grad_dir, v_to_q(self._sim.get_agent_state().rotation)
                    )
                    < alpha
                )
                else SimulatorActions.RIGHT
            )
            self._reset_agent_state(current_state)

            # Check if forward reduces geodesic distance
            curr_dist = self._geo_dist(goal_pos)
            self._sim.step(SIM_NAME_TO_ACTION[SimulatorActions.FORWARD.value])
            new_dist = self._geo_dist(goal_pos)
            new_pos = self._sim.get_agent_state().position
            movement_size = np.sqrt(
                np.sum(np.square(new_pos - current_state.position))
            )
            self._reset_agent_state(current_state)
            if new_dist < curr_dist and movement_size / self._step_size > 0.99:
                return self._get_return_value(SimulatorActions.FORWARD)
            else:
                return self._get_return_value(best_turn)

    def _reset_agent_state(self, state: habitat_sim.AgentState) -> None:
        self._sim.set_agent_state(
            state.position, state.rotation, reset_sensors=False
        )

    @staticmethod
    def _angle_between_quats(q1: np.quaternion, q2: np.quaternion) -> float:
        q1_inv = np.conjugate(q1)
        dq = quaternion.as_float_array(q1_inv * q2)

        return 2 * np.arctan2(np.linalg.norm(dq[1:]), np.abs(dq[0]))

    @staticmethod
    def _quat_from_two_vectors(v0: np.array, v1: np.array) -> np.quaternion:
        v0 = v0 / np.linalg.norm(v0)
        v1 = v1 / np.linalg.norm(v1)
        c = v0.dot(v1)
        if c < (-1 + 1e-8):
            c = max(c, -1)
            m = np.stack([v0, v1], 0)
            _, _, vh = np.linalg.svd(m, full_matrices=True)
            axis = vh.T[:, 2]
            w2 = (1 + c) * 0.5
            w = np.sqrt(w2)
            axis = axis * np.sqrt(1 - w2)
            return np.quaternion(w, *axis)

        axis = np.cross(v0, v1)
        s = np.sqrt((1 + c) * 2)
        return np.quaternion(s * 0.5, *(axis / s))

    def _geo_dist(self, goal_pos: np.array) -> float:
        return self._sim.geodesic_distance(
            self._sim.get_agent_state().position, goal_pos
        )

    def _est_max_grad_dir(self, goal_pos: np.array) -> np.array:

        current_pos = self._sim.get_agent_state().position

        if self.mode == "spath":
            points = self._sim.straight_spath_points(
                self._sim.get_agent_state().position, goal_pos
            )
            # NB: Add a little offset as things get weird if
            # points[1] - points[0] is anti-parallel with forward
            if len(points) < 2:
                return None
            max_grad_dir = self._quat_from_two_vectors(
                self._sim.FORWARD,
                points[1]
                - points[0]
                + 1e-3 * np.cross(self._sim.UP, self._sim.FORWARD),
            )
        else:
            current_rotation = self._sim.get_agent_state().rotation
            current_dist = self._geo_dist(goal_pos)

            best_delta = -1e10
            best_rotation = current_rotation
            for _ in range(0, 360, self._sim.config.TURN_ANGLE):
                sim_action = SIM_NAME_TO_ACTION[SimulatorActions.FORWARD.value]
                self._sim.step(sim_action)
                new_delta = current_dist - self._geo_dist(goal_pos)

                if new_delta > best_delta:
                    best_rotation = self._sim.get_agent_state().rotation
                    best_delta = new_delta

                # If the best delta is within (1 - cos(TURN_ANGLE))% of the best
                # delta (the step size), then we almost certianly have found the
                # max grad dir and should just exit
                if np.isclose(
                    best_delta,
                    self._max_delta,
                    rtol=1 - np.cos(np.deg2rad(self._sim.config.TURN_ANGLE)),
                ):
                    break

                self._sim.set_agent_state(
                    current_pos,
                    self._sim.get_agent_state().rotation,
                    reset_sensors=False,
                )

                sim_action = SIM_NAME_TO_ACTION[SimulatorActions.LEFT.value]
                self._sim.step(sim_action)

            self._sim.set_agent_state(
                current_pos, current_rotation, reset_sensors=False
            )

            max_grad_dir = v_to_q(best_rotation)

        return max_grad_dir

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        assert new_mode in {"spath", "grad"}
        if new_mode == "spath":
            assert (
                getattr(self._sim, "straight_spath_points", None) is not None
            )
        self._mode = new_mode

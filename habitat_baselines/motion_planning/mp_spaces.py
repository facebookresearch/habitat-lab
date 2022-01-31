import os
import os.path as osp
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
from PIL import Image

from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import IkHelper
from habitat_baselines.motion_planning.robot_target import RobotTarget

try:
    from ompl import base as ob  # pylint: disable=import-error
    from ompl import geometric as og  # pylint: disable=import-error
except ImportError:
    pass


def to_ob_state(vec: np.ndarray, space: "ob.StateSpace", dim: int):
    ob_vec = ob.State(space)
    for i in range(dim):
        ob_vec[i] = vec[i]
    return ob_vec


class MpSpace(ABC):
    """
    Defines an abstract planning space for OMPL to interface with.
    """

    def __init__(self, use_sim: RearrangeSim, ik: IkHelper):
        self._mp_sim = use_sim
        self._ik = ik

    @abstractmethod
    def convert_state(self, x: Iterable) -> np.ndarray:
        pass

    @abstractmethod
    def set_arm(self, x: Union[List[float], np.ndarray]):
        pass

    def set_env_state(self, env_state: Dict[str, Any]):
        self.env_state = env_state

    @abstractmethod
    def get_range(self) -> float:
        """
        Gets the planner step size range.
        """

    @abstractmethod
    def get_state_lims(self, restrictive: bool = False) -> np.ndarray:
        """
        Get the state limits of the planning problem.
        """

    @abstractmethod
    def get_state_dim(self) -> int:
        """
        Get the dimensionality of the planning problem
        """

    @abstractmethod
    def get_start_goal(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets the used start and goal states for the planner. This is after
        clipping and any additional pre-processing.
        """

    @abstractmethod
    def convert_sol(self, path) -> np.ndarray:
        """
        Convert a solution from OMPL format to numpy array
        """

    @abstractmethod
    def get_planner(self, si: "ob.SpaceInformation"):
        pass

    @abstractmethod
    def set_problem(
        self,
        pdef: "ob.ProblemDefinition",
        space: "ob.StateSpace",
        si: "ob.SpaceInformation",
        start_state: "ob.State",
        targ_state: RobotTarget,
    ):
        """
        Sets up the planning problem
        """

    def render_start_targ(
        self,
        render_dir: str,
        subdir: str,
        targ_state: np.ndarray,
        suffix: str = "targ",
    ):
        """
        Renders the start and target to images for visualization
        """


def getPathLengthObjWithCostToGo(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))
    return obj


class JsMpSpace(MpSpace):
    def __init__(self, use_sim, ik, start_num_calls, should_render):
        super().__init__(use_sim, ik)
        # self._lower_joint_lims, self._upper_joint_lims = self._ik.get_joint_limits()
        joint_lims = self.get_state_lims(True)
        self._lower_joint_lims, self._upper_joint_lims = (
            joint_lims[:, 0],
            joint_lims[:, 1],
        )
        self.num_calls = start_num_calls
        self._should_render = should_render

    def convert_state(self, x):
        return np.array([x[i] for i in range(7)])

    def _norm_joint_angle(self, angles):
        return np.arctan2(np.sin(angles), np.cos(angles))

    def get_planner(self, si):
        return og.RRTConnect(si)

    def get_state_lims(self, restrictive=False):
        """Get the state limits of the planning problem. If restrictive is true then
        this returns the joint limts based on the PyBullet joint limits
        """
        if restrictive:
            lower_joint_lims, upper_joint_lims = self._ik.get_joint_limits()
            lower_joint_lims = [
                -np.pi if np.isclose(a, 0.0) else a for a in lower_joint_lims
            ]
            upper_joint_lims = [
                np.pi if np.isclose(a, 2 * np.pi) else a
                for a in upper_joint_lims
            ]
            lower_joint_lims = self._norm_joint_angle(lower_joint_lims)
            upper_joint_lims = self._norm_joint_angle(upper_joint_lims)
            return np.stack([lower_joint_lims, upper_joint_lims], axis=-1)
        else:
            return np.stack([[-2 * np.pi] * 7, [2 * np.pi] * 7], axis=-1)

    def get_state_dim(self):
        return len(self._mp_sim._sim.robot.arm_joint_pos)

    def _fk(self, joints):
        """Sets the joint state and applys the change"""
        self._mp_sim.set_arm_pos(joints)
        self._mp_sim.micro_step()

    def get_start_goal(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.used_js_start, self.used_js_goal)

    def set_problem(
        self,
        pdef,
        space,
        si,
        js_start,
        robot_targ,
    ):
        """
        Sets up the OMPL problem
        """
        js_end = robot_targ.joints_target

        joint_shape = self._lower_joint_lims.shape

        js_start = self._norm_joint_angle(js_start)
        js_end = self._norm_joint_angle(js_end)

        # In case you want some padding to the limits for extra safety
        eps = np.full(joint_shape, 0.000)
        js_start = np.clip(
            js_start,
            self._lower_joint_lims + eps,
            self._upper_joint_lims - eps,
        )
        js_end = np.clip(
            js_end, self._lower_joint_lims + eps, self._upper_joint_lims - eps
        )

        self.used_js_start = js_start
        self.used_js_goal = js_end
        self.num_calls += 1

        js_start = to_ob_state(js_start, space, self.get_state_dim())
        js_end = to_ob_state(js_end, space, self.get_state_dim())

        def admiss_heuristic(cur_state, goal):
            use_cur_state = self.convert_state(cur_state)
            # FK to get both in EE space.
            self._fk(use_cur_state)
            cur_ee_state = self._mp_sim.get_ee_pos()
            ret = np.linalg.norm(robot_targ.ee_target_pos - cur_ee_state)
            return ret

        def getPathLengthObjWithCostToGo(si):
            obj = ob.PathLengthOptimizationObjective(si)
            obj.setCostToGoHeuristic(ob.CostToGoHeuristic(admiss_heuristic))
            return obj

        pdef.setStartAndGoalStates(js_start, js_end)
        pdef.setOptimizationObjective(getPathLengthObjWithCostToGo(si))

    def render_start_targ(self, render_dir, subdir, targ_state, suffix="targ"):
        if targ_state is not None:
            targ_viz_id = self._mp_sim.add_sphere(0.06, color=[0, 0, 1, 1])
            self._mp_sim.set_position(targ_state, targ_viz_id)

        use_dir = osp.join(render_dir, subdir)
        os.makedirs(use_dir, exist_ok=True)
        # Visualize the target position.
        # NOTE: The object will not immediately snap to the robot's hand if a target joint
        # state is provided. This is not an issue, it only affects this one
        # rendering.
        self._fk(self.used_js_goal)
        Image.fromarray(self._mp_sim.render()).save(
            osp.join(use_dir, f"{suffix}_goal_{self.num_calls}.jpeg")
        )

        self._fk(self.used_js_start)
        save_f_name = osp.join(
            use_dir, f"{suffix}_start_{self.num_calls}.jpeg"
        )
        Image.fromarray(self._mp_sim.render()).save(save_f_name)
        print("Rendered start / goal MP to ", save_f_name)
        if targ_state is not None:
            self._mp_sim.remove_object(targ_viz_id)

    def get_range(self):
        return 0.1

    def set_arm(self, des_joint_pos):
        des_joint_pos = self.convert_state(des_joint_pos)
        self._fk(des_joint_pos)
        self._mp_sim.set_state(self.env_state)
        des_joint_pos = np.array(des_joint_pos)[:7]

    def convert_sol(self, path):
        plan = np.array([self.convert_state(x) for x in path.getStates()])
        return plan

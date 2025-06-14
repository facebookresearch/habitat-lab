#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import magnum as mn
import numpy as np
import roboticstoolbox as rtb
from pydrake.solvers import MathematicalProgram, Solve
from spatialmath import SE3, Quaternion


class MathematicalProgramHelper:
    """This is similar to inverse_kinematics below, except:
    1. simplified solve: removed per-axis velocity constraints
    2. use a persistent MathematicalProgram
    """

    def __init__(self, robot, epsilon=1.0):
        self.robot = robot
        self.epsilon = epsilon
        self.dt = 0.03

        self.max_qd = np.array([2.1750] * 4 + [2.61] * 3)
        self.max_q = np.array(
            [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]
        )
        self.min_q = np.array(
            [-2.7437, -1.7837, -2.9007, -2.9, -2.8065, 0.5445, -3.0159]
        )
        self.q0 = (self.max_q + self.min_q) / 2
        self.K = np.array([10.0] * 7)

        self._build_program()

    def _build_program(self):
        self.prog = MathematicalProgram()
        self.qd = self.prog.NewContinuousVariables(7, "qd")

        # Fixed joint velocity bounds
        for i in range(7):
            self.prog.AddBoundingBoxConstraint(
                -self.max_qd[i], self.max_qd[i], self.qd[i]
            )

        # Time-varying joint limit constraints (qd bounds to stay within q limits)
        self.qd_limit_constraints = []
        for i in range(7):
            constraint = self.prog.AddBoundingBoxConstraint(
                0.0, 0.0, self.qd[i]
            )
            self.qd_limit_constraints.append(constraint)

        # Add a dummy cost we will overwrite each frame
        self.last_cost_handle = self.prog.AddCost(self.qd[0] ** 2)

    def solve(self, pose, q):
        if len(q) != 7:
            raise ValueError("q must be of length 7")
        self.robot.q = q

        J = self.robot.jacobe(q)
        v, _ = rtb.p_servo(
            self.robot.fkine(q, end=self.robot.links[8]), pose, 5
        )

        # Update position limit constraints per joint
        for i in range(7):
            upper = min((self.max_q[i] - q[i]) / self.dt, self.max_qd[i])
            lower = max((self.min_q[i] - q[i]) / self.dt, -self.max_qd[i])
            self.qd_limit_constraints[i].evaluator().set_bounds(
                [lower], [upper]
            )

        # Build the cost expression
        error = J @ self.qd - v
        nullspace_proj = np.eye(7) - np.linalg.pinv(J) @ J
        joint_centering = self.epsilon * (
            nullspace_proj @ (self.qd - self.K * (self.q0 - q))
        )
        total_cost = (
            error @ error
            + 0.05 * (self.qd @ self.qd)
            + joint_centering @ joint_centering
        )

        # Replace previous cost with updated one
        self.prog.RemoveCost(self.last_cost_handle)
        self.last_cost_handle = self.prog.AddCost(total_cost)

        result = Solve(self.prog)
        if result.is_success():
            return result.GetSolution(self.qd) * self.dt + q
        else:
            # print("Inverse kinematics failed")
            return q


def to_ik_pose(
    legacy_tuple: Optional[tuple[mn.Vector3, mn.Quaternion]]
) -> str:
    """
    Converts a transformation into the expected type (spatialmath) and conventions for the IK solver.
    """

    if legacy_tuple is None:
        return None

    v = legacy_tuple[0]
    q = legacy_tuple[1]

    if v is None or q is None:
        return None

    # apply a corrective rotation to the local frame in order to align the palm
    r = mn.Quaternion.rotation(-mn.Rad(0), mn.Vector3(0, 0, 1))
    q = q * r

    R = (
        (Quaternion([q.scalar, q.vector[0], q.vector[1], q.vector[2]]))
        .unit()
        .SE3()
    )
    t = SE3([v.x, v.y, v.z])
    pose = t * R

    return pose


class DifferentialInverseKinematics:
    def __init__(self):
        self.robot = rtb.models.Panda()

        # Change epsilon to change weightage of secondary task
        # 0 to have no secondary tasks
        self.epsilon = 0.1

        self._helper = MathematicalProgramHelper(self.robot, self.epsilon)

    def get_ee_T(self):
        """
        Get the end effector transform in robot base space as a Matrix4.
        """
        return mn.Matrix4(
            self.robot.fkine(self.robot.q, end=self.robot.links[8]).A
        )

    def inverse_kinematics(self, pose, q):
        if self._helper:
            return self._helper.solve(pose, q)

        if len(q) != 7:
            raise ValueError("q must be of length 7")

        self.robot.q = q

        jacobian = self.robot.jacobe(self.robot.q)

        v, _ = rtb.p_servo(
            self.robot.fkine(self.robot.q, end=self.robot.links[8]), pose, 5
        )

        # Create a MathematicalProgram
        prog = MathematicalProgram()

        # Create a joint velocity decision variable
        qd = prog.NewContinuousVariables(7, "qd")

        max_qd = [2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61]

        max_q = [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]
        min_q = [-2.7437, -1.7837, -2.9007, -2.9, -2.8065, 0.5445, -3.0159]

        # max_qdd = [15, 7.5, 10, 12.5, 15, 20, 20]

        # Add constraints to limit the joint velocities
        for _i in range(7):
            prog.AddConstraint(qd[_i] <= max_qd[_i])
            prog.AddConstraint(qd[_i] >= -max_qd[_i])
            prog.AddConstraint(self.robot.q[_i] + qd[_i] * 0.03 <= max_q[_i])
            prog.AddConstraint(self.robot.q[_i] + qd[_i] * 0.03 >= min_q[_i])

        # Add a cost to minimize the squared joint velocities
        error = jacobian @ qd - v

        K = [10, 10, 10, 10, 10, 10, 10]
        q0 = [(max_q[i] - min_q[i]) / 2 for i in range(7)]

        # Do joint centering in the null space
        joint_centering = self.epsilon * (
            (np.eye(7) - np.linalg.pinv(jacobian) @ jacobian)
            @ (qd - K * (np.array(q0) - self.robot.q))
        )
        prog.AddCost(
            error.dot(error)
            + 0.05 * qd.dot(qd)
            + joint_centering.dot(joint_centering)
        )

        alpha = prog.NewContinuousVariables(6, "alpha")
        prog.AddCost(-sum(alpha))
        for i in range(6):
            prog.AddLinearConstraint(alpha[i] >= 0.01)
            prog.AddLinearConstraint(alpha[i] <= 1)
            prog.AddLinearConstraint(alpha[i] * v[i] == (jacobian @ qd)[i])

        result = Solve(prog)
        if result.is_success():
            return result.GetSolution(qd) / 30 + self.robot.q
        else:
            # print("Inverse kinematics failed")
            return self.robot.q

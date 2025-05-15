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

    def get_ee_T(self):
        """
        Get the end effector transform in robot base space as a Matrix4.
        """
        return mn.Matrix4(self.robot.fkine(self.robot.q).A)

    def inverse_kinematics(self, pose, q):
        if len(q) != 7:
            raise ValueError("q must be of length 7")

        self.robot.q = q

        jacobian = self.robot.jacobe(self.robot.q)

        v, _ = rtb.p_servo(
            self.robot.fkine(self.robot.q, end=self.robot.links[7]), pose, 5
        )

        # Create a MathematicalProgram
        prog = MathematicalProgram()

        # Create a joint velocity decision variable
        qd = prog.NewContinuousVariables(7, "qd")

        max_qd = [2.1750, 2.1750, 2.1750, 2.1750, 2.61, 2.61, 2.61]
        max_q = [2.8973, 1.7628, 2.8973, -0.7315114, 2.8973, 3.7525, 2.8973]
        min_q = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        # max_qdd = [15, 7.5, 10, 12.5, 15, 20, 20]

        # Add constraints to limit the joint velocities
        # for _i in range(7):
        #     prog.AddConstraint(qd[_i] <= max_qd[_i])
        #     prog.AddConstraint(qd[_i] >= -max_qd[_i])
        #     prog.AddConstraint(self.robot.q[_i] + qd[_i] * 0.03 <= max_q[_i])
        #     prog.AddConstraint(self.robot.q[_i] + qd[_i] * 0.03 >= min_q[_i])

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
            # + joint_centering.dot(joint_centering)
        )

        # alpha = prog.NewContinuousVariables(6, "alpha")
        # prog.AddCost(-sum(alpha))
        # for i in range(6):
        #     prog.AddLinearConstraint(alpha[i] >= 0.01)
        #     prog.AddLinearConstraint(alpha[i] <= 1)
        #     prog.AddLinearConstraint(alpha[i] * v[i] == (jacobian @ qd)[i])

        result = Solve(prog)
        if result.is_success():
            return result.GetSolution(qd) / 30 + self.robot.q
        else:
            print("Inverse kinematics failed")
            return self.robot.q

from pydrake.solvers import MathematicalProgram, Solve
from time import time
import logging 
import numpy as np
import roboticstoolbox as rtb

class DifferentialInverseKinematics:

    def __init__(self):
        
        self.robot = rtb.models.Panda()

        # Change epsilon to change weightage of secondary task
        # 0 to have no secondary taskss
        self.epsilon = 0.1

    def inverse_kinematics(self, pose, q):

        if len(q) != 7:
            raise ValueError("q must be of length 7")
        
        self.robot.q = q

        jacobian = self.robot.jacobe(self.robot.q)

        v, _ = rtb.p_servo(self.robot.fkine(self.robot.q), pose, 5)

        # Create a MathematicalProgram
        prog = MathematicalProgram()

        # Create a joint velocity decision variable
        qd = prog.NewContinuousVariables(7, "qd")

        max_qd = [2.1750 , 2.1750 , 2.1750 , 2.1750 , 2.61, 2.61, 2.61]
        max_q = [2.8973, 1.7628, 2.8973, -0.68315114, 2.8973, 3.7525, 2.8973]
        min_q = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        max_qdd = [15, 7.5, 10, 12.5, 15, 20, 20]

        # Add constraints to limit the joint velocities
        for i in range(7):
            # prog.AddConstraint(qd[i] <= max_qd[i])
            # prog.AddConstraint(qd[i] >= -max_qd[i])
            # prog.AddConstraint(self.robot.q[i]+qd[i]*0.05 <= max_q[i])
            # prog.AddConstraint(self.robot.q[i]+qd[i]*0.05 >= min_q[i])
            # prog.AddConstraint(qd[i]-self.robot.qd[i] <= max_qdd[i]*0.05)
            # prog.AddConstraint(qd[i]-self.robot.qd[i] >= -max_qdd[i]*0.05)
            pass

        # Add a cost to minimize the squared joint velocities
        error = jacobian@qd - v
        
        K = [10, 10, 10, 10, 10, 10, 10]
        q0 = [(max_q[i]-min_q[i])/2 for i in range(7)]

        # Do joint centering in the null space
        joint_centering = self.epsilon*( (np.eye(7) - np.linalg.pinv(jacobian)@jacobian)@(qd - K*( np.array(q0) - self.robot.q ) ) )
        prog.AddCost(error.dot(error) + 0.05*qd.dot(qd) +  joint_centering.dot(joint_centering))



        alpha = prog.NewContinuousVariables(6, "alpha")
        prog.AddCost(-sum(alpha))
        for i in range(6):
            prog.AddLinearConstraint(alpha[i]>=0.01)
            prog.AddLinearConstraint(alpha[i]<=1)
            prog.AddLinearConstraint(alpha[i]*v[i]==(jacobian@qd)[i])
        

        result = Solve(prog)
        if result.is_success():
            
            return result.GetSolution(qd)/30 + self.robot.q
        else:
            
            return self.robot.q
        
        









    

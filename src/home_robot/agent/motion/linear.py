import numpy as np
from home_robot.agent.motion.base import Planner
from home_robot.agent.motion.robot import STRETCH_STANDOFF_WITH_MARGIN
from home_robot.agent.motion.robot import HelloStretchIdx


class LinearPlanner(Planner):
    """linear configuration-space plans"""

    def __init__(self, robot, step_size=0.1, *args, **kwargs):
        super(LinearPlanner, self).__init__(robot, *args, **kwargs)
        self.step_size = step_size

    def plan(self, q0, qg):
        """linear planner goes from one point to another."""
        # interpolate to qg and test
        # traj = [q for q in self.robot.interpolate(q0, qg)]
        traj = []
        ts = []
        t = 0
        traj.append(q0)
        ts.append(0.0)
        for q in self.robot.interpolate(q0, qg):
            if self.robot.validate(q):
                traj.append(q)
                t = t + 0.1
                ts.append(t)
            else:
                return None
        return traj, ts


class StretchLinearWithOffsetPlanner(Planner):
    """Plan to a stand-off position from which you can move straight to the goal grasp position"""

    def __init__(
        self,
        robot,
        step_size=0.1,
        standoff_range=[
            STRETCH_STANDOFF_WITH_MARGIN,
            STRETCH_STANDOFF_WITH_MARGIN + 0.2,
        ],
        *args,
        **kwargs
    ):
        """save the standoff distance so we can randomly sample one that works"""
        super(StretchLinearWithOffsetPlanner, self).__init__(robot, *args, **kwargs)
        self.step_size = step_size
        self.standoff_range = standoff_range
        self.standoff_min = self.standoff_range[0]
        self.standoff_rng = self.standoff_range[1] - self.standoff_range[0]

    def _interpolate(self, q0, q1, ignored=[]):
        t = 0
        traj, ts = [], []
        traj.append(q0)
        ts.append(t)
        for q in self.robot.interpolate(q0, q1):
            if self.robot.validate(q, ignored=ignored):
                traj.append(q)
                t += 0.1
                ts.append(t)
            else:
                return None
        return traj, ts

    def plan(self, q0, poses, grasp=True, tries=100, ignored=[]):
        """we assume that the arm has to be at least extended enough that we can do this, so check
        the arm extension. This planner is designed for work with the stretch only.
        """
        # darm = qg[HelloStretchIdx.ARM] - q0[HelloStretchIdx.ARM]
        # retract arm first
        if not self.robot.validate(q0):
            raise RuntimeError("invalid start configuration for planner")
        q_retract = q0.copy()
        # TODO: for debugging
        # self.robot.validate(qi)
        # input('---')
        q_retract[HelloStretchIdx.ARM] = 0
        sequences = []
        retract = self._interpolate(q0, q_retract)
        # Check to see if planning failed here
        if retract is None:
            return None
        sequences.append(retract)

        # Try to find IK solutions
        #
        for i in range(tries):
            idx = i % len(poses)
            pos, orn = poses[idx]
            q = self.robot.ik((pos, orn), self.robot.sample_uniform(q0, pos))
            # Now move it back
            # This function puts the robot farther away by some random distance
            q = self.robot.extend_arm_to(
                q, (np.random.random() * self.standoff_rng) + self.standoff_min
            )
            # print(q[HelloStretchIdx.ARM], self.robot.validate(q), self.robot.fk())
            # input('---')
            # TODO: to debug this, pass verbose=true to see what it's colliding with
            if not self.robot.validate(q, ignored=ignored, verbose=False):
                # Tried to compute a standoff at arm = 0
                continue
            # input('good solution')
            q_grasp = q

            # Move to standoff pose with arm retracted
            q_standoff = q_grasp.copy()
            q_standoff[HelloStretchIdx.ARM] = 0
            standoff = self._interpolate(q_retract, q_standoff)
            if standoff is None:
                # input('failed standoff')
                continue

            # Move to grasp - extend the arm
            grasp = self._interpolate(q_standoff, q_grasp, ignored=ignored)
            if grasp is not None:
                sequences.append(standoff)
                sequences.append(grasp)
                break
            else:
                # Grasp appraoch fails
                # input('failed grasp')
                continue

        # Move to grasp - combine all the different sequences
        traj, ts = [], []
        for _traj, _ts in sequences:
            traj += _traj
            ts += _ts
        return traj, ts


class StretchLinearIKPlanner(Planner):
    """Plan to a stand-off position from which you can move straight to the goal grasp position
    This version of the planner uses base rotation and IK to figure out an approach that we can follow"""

    def __init__(
        self,
        robot,
        step_size=0.1,
        standoff_range=[
            STRETCH_STANDOFF_WITH_MARGIN,
            STRETCH_STANDOFF_WITH_MARGIN + 0.2,
        ],
        *args,
        **kwargs
    ):
        """save the standoff distance so we can randomly sample one that works"""
        super(StretchLinearWithOffsetPlanner, self).__init__(robot, *args, **kwargs)
        self.step_size = step_size
        self.standoff_range = standoff_range
        self.standoff_min = self.standoff_range[0]
        self.standoff_rng = self.standoff_range[1] - self.standoff_range[0]

    def _interpolate(self, q0, q1, ignored=[]):
        t = 0
        traj, ts = [], []
        traj.append(q0)
        ts.append(t)
        for q in self.robot.interpolate(q0, q1):
            if self.robot.validate(q, ignored=ignored):
                traj.append(q)
                t += 0.1
                ts.append(t)
            else:
                return None
        return traj, ts

    def plan(self, q0, poses, tries=100, ignored=[]):
        """we assume that the arm has to be at least extended enough that we can do this, so check
        the arm extension. This planner is designed for work with the stretch only.
        """
        # darm = qg[HelloStretchIdx.ARM] - q0[HelloStretchIdx.ARM]
        # retract arm first
        if not self.robot.validate(q0):
            raise RuntimeError("invalid start configuration for planner")
        # Start from position with the arm fully retracted
        q_retract = q0.copy()
        q_retract[HelloStretchIdx.ARM] = 0
        sequences = []
        retract = self._interpolate(q0, q_retract)
        # Check to see if planning failed here
        if retract is None:
            return None
        sequences.append(retract)

        # Try to find IK solutions
        #
        for i in range(tries):
            idx = i % len(poses)
            pos, orn = poses[idx]
            q = self.robot.ik((pos, orn), self.robot.sample_uniform(q0, pos))
            # Now move it back
            # This function puts the robot farther away by some random distance
            q = self.robot.extend_arm_to(
                q, (np.random.random() * self.standoff_rng) + self.standoff_min
            )
            # if grasping... make sure the gripper is open
            if grasp:
                q = self.robot.config_open_gripper(q)
            # print(q[HelloStretchIdx.ARM], self.robot.validate(q), self.robot.fk())
            # input('---')
            # TODO: to debug this, pass verbose=true to see what it's colliding with
            # if not self.robot.validate(q, ignored=ignored, verbose=False):
            if not self.robot.validate(q, verbose=False):
                # Tried to compute a standoff at arm = 0
                continue
            # input('good solution')
            q_grasp = q

            # Move to standoff pose with arm retracted
            q_standoff = self.robot.config_open_gripper(q_grasp.copy())
            q_standoff[HelloStretchIdx.ARM] = 0
            standoff = self._interpolate(q_retract, q_standoff)
            if standoff is None:
                # input('failed standoff')
                continue

            # Move to grasp - extend the arm
            # grasp = self._interpolate(q_standoff, q_grasp, ignored=ignored)
            grasp = self._interpolate(q_standoff, q_grasp)
            if grasp is not None:
                sequences.append(standoff)
                sequences.append(grasp)
                break
            else:
                # Grasp appraoch fails
                # input('failed grasp')
                continue

        # Move to grasp - combine all the different sequences
        traj, ts = [], []
        for _traj, _ts in sequences:
            traj += _traj
            ts += _ts
        return traj, ts

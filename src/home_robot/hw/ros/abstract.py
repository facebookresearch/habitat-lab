import abc
import numpy as np
import time
import timeit
import trimesh.transformations as tra


from home_robot.agent.motion.robot import HelloStretch, HelloStretchIdx
from home_robot.agent.motion.robot import (
    STRETCH_HOME_Q,
    STRETCH_GRASP_FRAME,
    STRETCH_STANDOFF_DISTANCE,
)

# For handling grasping
from home_robot.utils.pose import to_pos_quat


BASE_X_IDX = HelloStretchIdx.BASE_X
BASE_Y_IDX = HelloStretchIdx.BASE_Y
BASE_THETA_IDX = HelloStretchIdx.BASE_THETA
LIFT_IDX = HelloStretchIdx.LIFT
ARM_IDX = HelloStretchIdx.ARM

# from home_robot.agent.motion.robot import (
#        BASE_X_IDX, B
GRIPPER_IDX = HelloStretchIdx.GRIPPER
WRIST_ROLL_IDX = HelloStretchIdx.WRIST_ROLL
WRIST_PITCH_IDX = HelloStretchIdx.WRIST_PITCH
WRIST_YAW_IDX = HelloStretchIdx.WRIST_YAW

# Head stuff
HEAD_PAN_IDX = HelloStretchIdx.HEAD_PAN
HEAD_TILT_IDX = HelloStretchIdx.HEAD_TILT

# Home config from main file
HOME = STRETCH_HOME_Q


class AbstractStretchInterface(abc.ABC):
    """Basic abstract class containing references to methods we need to overwrite"""

    def __init__(self):
        pass

    def reset_state(self):
        self.pos = np.zeros(self.dof)
        self.vel = np.zeros(self.dof)
        self.frc = np.zeros(self.dof)

    def get_model(self):
        return self.model

    def get_backend(self):
        """reference to planner physics objects"""
        return self.obj.get_backend()

    def goto(self, q, *args, **kwargs):
        raise NotImplementedError

    def wait(self, q1, max_wait_t=10.0, no_base=False, verbose=False):
        """helper function to wait until we reach a position"""
        t0 = timeit.default_timer()
        while (timeit.default_timer() - t0) < max_wait_t:
            # update and get pose metrics
            q0, dq0 = self.update()
            err = np.abs(q1 - q0)
            if no_base:
                err[:3] = 0.0
            dt = timeit.default_timer() - t0
            if verbose:
                print("goal =", q1)
                print(dt, err < self.exec_tol)
                self.pretty_print(err)
            if np.all(err < self.exec_tol):
                return True
            time.sleep(self.wait_time_step)
        return False

    def set_planner_config(self, q):
        """update planner representation internally"""
        self.model.set_config(q)

    def pretty_print(self, q):
        print("-" * 20)
        print("lift:      ", q[LIFT_IDX])
        print("arm:       ", q[ARM_IDX])
        print("gripper:   ", q[GRIPPER_IDX])
        print("wrist yaw: ", q[WRIST_YAW_IDX])
        print("wrist pitch:", q[WRIST_PITCH_IDX])
        print("wrist roll: ", q[WRIST_ROLL_IDX])
        print("head pan:   ", q[HEAD_PAN_IDX])
        print("head tilt:   ", q[HEAD_TILT_IDX])
        print("-" * 20)

    def look_at_ee(self, wait=False):
        q, _ = self.update()
        q = self.model.update_look_at_ee(q)
        self.goto(q, wait=wait, move_base=False)

    def look_front(self, wait=False):
        q, _ = self.update()
        q = self.model.update_look_front(q)
        self.goto(q, wait=wait, move_base=False)

    def look_ahead(self, wait=False):
        q, _ = self.update()
        q = self.model.update_look_ahead(q)
        self.goto(q, wait=wait, move_base=False)

    def stow(self, wait=False):
        """put that wrist away so we dont break it"""
        q = STRETCH_HOME_Q
        self.goto(q, wait=wait, move_base=False)

    def stow_wrist(self, wait=False):
        """put that wrist away so we dont break it"""
        q, _ = self.update()
        q[HelloStretchIdx.WRIST_ROLL] = STRETCH_HOME_Q[HelloStretchIdx.WRIST_ROLL]
        q[HelloStretchIdx.WRIST_PITCH] = STRETCH_HOME_Q[HelloStretchIdx.WRIST_PITCH]
        q[HelloStretchIdx.WRIST_YAW] = STRETCH_HOME_Q[HelloStretchIdx.WRIST_YAW]
        self.goto(q, wait=wait, move_base=False)

    def fk(self, q=None, link_name=None):
        if link_name is None:
            link_name = STRETCH_GRASP_FRAME
        if q is None:
            q, _ = self.update()
        self.model.set_config(q)
        pos, rot = self.model.get_link_pose(link_name)
        x, y, z, w = rot
        pose = tra.quaternion_matrix([w, x, y, z])
        pose[:3, 3] = pos
        return pose

    def goto_static_grasp(self, grasps, scores=None, pause=False):
        """
        Go to a grasp position, given a list of acceptable grasps
        """
        if scores is None:
            scores = np.arange(len(grasps))
        q, _ = self.update()

        grasp_offset = np.eye(4)
        # Some magic numbers here
        # This should correct for the length of the Stretch gripper and the gripper upon which
        # Graspnet was trained
        grasp_offset[2, 3] = (-1 * STRETCH_STANDOFF_DISTANCE) + 0.12
        for i, grasp in enumerate(grasps):
            grasps[i] = grasp @ grasp_offset

        # q[:3] = np.zeros(3)
        for grasp, score in sorted(zip(grasps, scores), key=lambda p: p[1]):
            grasp_pose = to_pos_quat(grasp)
            qi = self.model.static_ik(grasp_pose, q)
            print("grasp xyz =", grasp_pose[0])
            if qi is not None:
                print(" - IK found")
                self.model.set_config(qi)
                input("---")
            else:
                # Grasp attempt failure
                continue
            # Record the initial q value here and use it
            theta0 = q[2]
            q1 = qi.copy()
            q1[HelloStretchIdx.LIFT] += 0.08
            # q1[HelloStretchIdx.LIFT] += 0.2
            if q1 is not None:
                # Run a validity check to make sure we can actually pick this thing up
                if not self.model.validate(q1):
                    print("invalid standoff config:", q1)
                    continue
                print("found standoff")
                q2 = qi
                # q2 = model.static_ik(grasp_pose, q1)
                if q2 is not None:
                    # if np.abs(eq1) < 0.075 and np.abs(eq2) < 0.075:
                    # go to the grasp and try it
                    q[HelloStretchIdx.LIFT] = 0.99
                    self.goto(q, move_base=False, wait=True, verbose=False)
                    if pause:
                        input("--> go high")
                    q_pre = q.copy()
                    q_pre[5:] = q1[5:]
                    q_pre = self.model.update_gripper(q_pre, open=True)
                    self.move_base(theta=q1[2])
                    time.sleep(2.0)
                    self.goto(q_pre, move_base=False, wait=False, verbose=False)
                    self.model.set_config(q1)
                    if pause:
                        input("--> gripper ready; go to standoff")
                    q1 = self.model.update_gripper(q1, open=True)
                    self.goto(q1, move_base=False, wait=True, verbose=False)
                    if pause:
                        input("--> go to grasp")
                    self.move_base(theta=q2[2])
                    time.sleep(2.0)
                    self.goto(q_pre, move_base=False, wait=False, verbose=False)
                    self.model.set_config(q2)
                    q2 = self.model.update_gripper(q2, open=True)
                    self.goto(q2, move_base=False, wait=True, verbose=True)
                    if pause:
                        input("--> close the gripper")
                    q2 = self.model.update_gripper(q2, open=False)
                    self.goto(q2, move_base=False, wait=False, verbose=True)
                    time.sleep(2.0)
                    q = self.model.update_gripper(q, open=False)
                    self.goto(q, move_base=False, wait=True, verbose=False)
                    self.move_base(theta=q[0])
                    return True
        return False

from pdb import post_mortem
import numpy as np
import os
import home_robot.utils.bullet as hrb
import pybullet as pb
import trimesh.transformations as tra

from tracikpy import TracIKSolver

from home_robot.agent.motion.space import Space
from home_robot.utils.pose import to_matrix


class Robot(object):
    """placeholder"""

    def __init__(self, name="robot", urdf_path=None, visualize=False, assets_path=None):
        # Load and create planner
        self.backend = hrb.PbClient(visualize=visualize)
        # Create object reference
        self.ref = self.backend.add_articulated_object(
            name, urdf_path, assets_path=assets_path
        )

    def get_backend(self):
        raise NotImplementedError

    def get_dof(self):
        """return degrees of freedom of the robot"""
        return self.dof

    def set_config(self, q):
        """put the robot in the right position"""
        raise NotImplementedError

    def get_config(self):
        """turn current state into a vector"""
        raise NotImplementedError

    def set_head_config(self, q):
        """just for the head"""
        raise NotImplementedError

    def set_camera_to_head(self, camera, q=None):
        """take a bullet camera and put it on the robot's head"""
        if q is not None:
            self.set_head_config(q)
        raise NotImplementedError


# Stretch stuff
DEFAULT_STRETCH_URDF = "assets/hab_stretch/urdf/stretch_dex_wrist_simplified.urdf"
# PLANNER_STRETCH_URDF =  'assets/hab_stretch/urdf/planner_stretch_dex_wrist_simplified.urdf'
PLANNER_STRETCH_URDF = "assets/hab_stretch/urdf/planner_calibrated.urdf"
STRETCH_HOME_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.2,  # lift
        0.057,  # arm
        0.0,  # gripper rpy
        0.0,
        0.0,
        3.0,  # wrist,
        0.0,
        0.0,
    ]
)
STRETCH_PREGRASP_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.6,  # lift
        0.06,  # arm
        0.0,  # gripper rpy
        3.14,  # wrist roll
        -1.57,  # wrist pitch
        0.0,  # wrist yaw
        0.0,
        0.0,
    ]
)

# This is the gripper, and the distance in the gripper frame to where the fingers will roughly meet
STRETCH_GRASP_FRAME = "link_straight_gripper"
STRETCH_STANDOFF_DISTANCE = 0.235
STRETCH_STANDOFF_WITH_MARGIN = 0.25
# Offset from a predicted grasp point to STRETCH_GRASP_FRAME
STRETCH_GRASP_OFFSET = np.eye(4)
STRETCH_GRASP_OFFSET[:3, 3] = np.array([0, 0, -1 * STRETCH_STANDOFF_DISTANCE])
# Offset from STRETCH_GRASP_FRAME to predicted grasp point
STRETCH_TO_GRASP = np.eye(4)
STRETCH_TO_GRASP[:3, 3] = np.array([0, 0, STRETCH_STANDOFF_DISTANCE])
STRETCH_GRIPPER_OPEN = 0.22
STRETCH_GRIPPER_CLOSE = -0.2
# STRETCH_GRIPPER_CLOSE = -0.5


class HelloStretchIdx:
    BASE_X = 0
    BASE_Y = 1
    BASE_THETA = 2
    LIFT = 3
    ARM = 4
    GRIPPER = 5
    WRIST_ROLL = 6
    WRIST_PITCH = 7
    WRIST_YAW = 8
    HEAD_PAN = 9
    HEAD_TILT = 10


class HelloStretch(Robot):
    """define motion planning structure for the robot"""

    # DEFAULT_BASE_HEIGHT = 0.09
    DEFAULT_BASE_HEIGHT = 0
    GRIPPER_OPEN = 0.6
    GRIPPER_CLOSED = -0.3

    default_step = np.array(
        [
            0.1,
            0.1,
            0.2,  # x y theta
            0.025,
            0.025,  # lift and arm
            0.3,  # gripper
            0.1,
            0.1,
            0.1,  # wrist rpy
            0.2,
            0.2,  # head
        ]
    )
    default_tols = np.array(
        [
            0.1,
            0.1,
            0.01,  # x y theta
            0.001,
            0.0025,  # lift and arm
            0.01,  # gripper
            0.01,
            0.01,
            0.01,  # wrist rpy
            10.0,
            10.0,  # head - TODO handle this better
        ]
    )
    # look_at_ee = np.array([-np.pi/2, -np.pi/8])
    look_at_ee = np.array([-np.pi / 2, -np.pi / 4])
    look_front = np.array([0.0, -np.pi / 4])
    look_ahead = np.array([0.0, 0.0])
    ee_link_name = "link_straight_gripper"

    def __init__(self, name="robot", urdf_path=None, visualize=False, root="."):
        """Create the robot in bullet for things like kinematics; extract information"""

        # urdf
        if urdf_path is None:
            urdf_path = PLANNER_STRETCH_URDF
        urdf_path = os.path.join(root, urdf_path)
        super(HelloStretch, self).__init__(name, urdf_path, visualize)
        # DOF: 3 for ee roll/pitch/yaw
        #      1 for gripper
        #      1 for ee extension
        #      3 for base x/y/theta
        #      2 for head
        self.dof = 3 + 2 + 4 + 2
        self.base_height = self.DEFAULT_BASE_HEIGHT
        self.ik_solver = TracIKSolver(
            urdf_path, "world", "link_straight_gripper", timeout=0.1, epsilon=1e-5
        )
        assert self.ik_solver.number_of_joints == 11
        self.static_ik_solver = TracIKSolver(
            urdf_path,
            "base_theta_link",
            "link_straight_gripper",
            timeout=0.1,
            epsilon=1e-5,
        )
        assert self.static_ik_solver.number_of_joints == 9

        self.lift_arm_ik_solver = TracIKSolver(
            urdf_path, "base_link", "link_arm_l0", timeout=0.1, epsilon=1e-5
        )
        assert self.lift_arm_ik_solver.number_of_joints == 5

        # ranges for joints
        self.range = np.zeros((self.dof, 2))

        # Create object reference
        self.set_pose = self.ref.set_pose
        self.set_joint_position = self.ref.set_joint_position

        self._update_joints()

    def set_head_config(self, q):
        # WARNING: this sets all configs
        bidxs = [HelloStretchIdx.HEAD_PAN, HelloStretchIdx.HEAD_TILT]
        bidxs = [self.joint_idx[b] for b in bidxs]
        if len(q) == self.dof:
            qidxs = bidxs
        elif len(q) == 2:
            qidxs = [0, 1]
        else:
            raise RuntimeError("unsupported number of head joints")

        for idx, qidx in zip(bidxs, qidxs):
            # Idx is the urdf joint index
            # qidx is the idx in the provided query
            qq = q[qidx]
            self.ref.set_joint_position(idx, qq)

    def sample_uniform(self, q0=None, pos=None, radius=2.0):
        """Sample random configurations to seed the ik planner"""
        q = (np.random.random(self.dof) * self._rngs) + self._mins
        q[HelloStretchIdx.BASE_THETA] = np.random.random() * np.pi * 2
        # Set the gripper state
        if q0 is not None:
            q[HelloStretchIdx.GRIPPER] = q0[HelloStretchIdx.GRIPPER]
        # Set the position to sample poses
        if pos is not None:
            x, y = pos[0], pos[1]
        elif q0 is not None:
            x = q0[HelloStretchIdx.BASE_X]
            y = q0[HelloStretchIdx.BASE_Y]
        else:
            x, y = None, None
        # Randomly sample
        if x is not None:
            theta = np.random.random() * 2 * np.pi
            dx = radius * np.cos(theta)
            dy = radius * np.sin(theta)
            q[HelloStretchIdx.BASE_X] = x + dx
            q[HelloStretchIdx.BASE_Y] = y + dy
        return q

    def config_open_gripper(self, q):
        q[HelloStretchIdx.GRIPPER] = self.range[HelloStretchIdx.GRIPPER][1]
        return q

    def config_close_gripper(self, q):
        q[HelloStretchIdx.GRIPPER] = self.range[HelloStretchIdx.GRIPPER][0]
        return q

    def _update_joints(self):
        """Get joint info from URDF or otherwise provide it"""
        self.joint_idx = [-1] * self.dof
        # Get the joint info we need from this
        joint_lift = self.ref.get_joint_info_by_name("joint_lift")
        self.range[HelloStretchIdx.LIFT] = np.array(
            [joint_lift.lower_limit, joint_lift.upper_limit]
        )
        self.joint_idx[HelloStretchIdx.LIFT] = joint_lift.index
        joint_head_pan = self.ref.get_joint_info_by_name("joint_head_pan")
        self.range[HelloStretchIdx.HEAD_PAN] = np.array(
            [joint_head_pan.lower_limit, joint_head_pan.upper_limit]
        )
        self.joint_idx[HelloStretchIdx.HEAD_PAN] = joint_head_pan.index
        joint_head_tilt = self.ref.get_joint_info_by_name("joint_head_tilt")
        self.range[HelloStretchIdx.HEAD_TILT] = np.array(
            [joint_head_tilt.lower_limit, joint_head_tilt.upper_limit]
        )
        self.joint_idx[HelloStretchIdx.HEAD_TILT] = joint_head_tilt.index
        joint_wrist_yaw = self.ref.get_joint_info_by_name("joint_wrist_yaw")
        self.range[HelloStretchIdx.WRIST_YAW] = np.array(
            [joint_wrist_yaw.lower_limit, joint_wrist_yaw.upper_limit]
        )
        self.joint_idx[HelloStretchIdx.WRIST_YAW] = joint_wrist_yaw.index
        joint_wrist_roll = self.ref.get_joint_info_by_name("joint_wrist_roll")
        self.range[HelloStretchIdx.WRIST_ROLL] = np.array(
            [joint_wrist_roll.lower_limit, joint_wrist_roll.upper_limit]
        )
        self.joint_idx[HelloStretchIdx.WRIST_ROLL] = joint_wrist_roll.index
        joint_wrist_pitch = self.ref.get_joint_info_by_name("joint_wrist_pitch")
        self.range[HelloStretchIdx.WRIST_PITCH] = np.array(
            [joint_wrist_pitch.lower_limit, joint_wrist_pitch.upper_limit]
        )
        self.joint_idx[HelloStretchIdx.WRIST_PITCH] = joint_wrist_pitch.index

        # arm position
        # TODO: fix this so that it is not hard-coded any more
        self.range[HelloStretchIdx.ARM] = np.array([0.0, 0.5])
        self.arm_idx = []
        upper_limit = 0
        for i in range(4):
            joint = self.ref.get_joint_info_by_name("joint_arm_l%d" % i)
            self.arm_idx.append(joint.index)
            # TODO remove debug statement
            # print(i, joint.name, joint.lower_limit, joint.upper_limit)
            upper_limit += joint.upper_limit

        # TODO: gripper
        self.gripper_idx = []
        for i in ["right", "left"]:
            joint = self.ref.get_joint_info_by_name("joint_gripper_finger_%s" % i)
            self.gripper_idx.append(joint.index)
            print(i, joint.name, joint.lower_limit, joint.upper_limit)
            self.range[HelloStretchIdx.GRIPPER] = (
                np.array([joint.lower_limit, joint.upper_limit]) * 0.5
            )

        self._mins = self.range[:, 0]
        self._maxs = self.range[:, 1]
        self._rngs = self.range[:, 1] - self.range[:, 0]

    def get_space(self) -> Space:
        """
        return space to sample configuration in
        """
        pass

    def get_backend(self):
        return self.backend

    def get_object(self) -> hrb.PbArticulatedObject:
        """return back-end reference to the Bullet object"""
        return self.ref

    def _set_joint_group(self, idxs, val):
        for idx in idxs:
            self.ref.set_joint_position(idx, val)

    def vanish(self):
        """get rid of the robot"""
        self.ref.set_pose([0, 0, 1000], [0, 0, 0, 1])

    def set_config(self, q):
        assert len(q) == self.dof
        x, y, theta = q[:3]
        # quaternion = pb.getQuaternionFromEuler((0, 0, theta))
        self.ref.set_pose((0, 0, self.base_height), [0, 0, 0, 1])
        # self.ref.set_pose((x, y, self.base_height), quaternion)
        self.ref.set_joint_position(0, x)
        self.ref.set_joint_position(1, y)
        self.ref.set_joint_position(2, theta)
        for idx, qq in zip(self.joint_idx, q):
            if idx < 0:
                continue
            self.ref.set_joint_position(idx, qq)
        # Finally set the arm and gripper as groups
        self._set_joint_group(self.arm_idx, q[HelloStretchIdx.ARM] / 4.0)
        self._set_joint_group(self.gripper_idx, q[HelloStretchIdx.GRIPPER])

    def plan_look_at(self, q0, xyz):
        """assume this is a relative xyz"""
        dx, dy = xyz[:2] - q0[:2]
        theta0 = q0[2]
        thetag = np.arctan2(dy, dx)
        action = np.zeros(self.dof)

        dist = np.abs(thetag - theta0)
        # Dumb check to see if we can get the rotation right
        if dist > np.pi:
            thetag -= np.pi / 2
            dist = np.abs(thetag - theta0)
        # Getting the direction right here
        dirn = 1.0 if thetag > theta0 else -1.0
        action[2] = dist * dirn

        look_action = q0.copy()
        self.update_look_ahead(look_action)

        # Compute the angle
        head_height = 1.2
        distance = np.linalg.norm([dx, dy])

        look_action[HelloStretchIdx.HEAD_TILT] = -1 * np.arctan(
            (head_height - xyz[2]) / distance
        )

        return [action, look_action]

    def interpolate(self, q0, qg, step=None, xy_tol=0.05, theta_tol=0.01):
        """interpolate from initial to final configuration. for this robot we break it up into
        four stages:
        1) rotate to point towards final location
        2) drive to final location
        3) rotate to final orientation
        4) move everything else
        """
        if step is None:
            step = self.default_step
        qi = q0.copy()
        theta0 = q0[HelloStretchIdx.BASE_THETA]
        thetag = qg[HelloStretchIdx.BASE_THETA]
        xy0 = q0[[HelloStretchIdx.BASE_X, HelloStretchIdx.BASE_Y]]
        xyg = qg[[HelloStretchIdx.BASE_X, HelloStretchIdx.BASE_Y]]
        dist = np.linalg.norm(xy0 - xyg)
        if dist > xy_tol:
            dx, dy = xyg - xy0
            theta = np.arctan2(dy, dx)
            for qi, ai in self.interpolate_angle(
                qi, theta0, theta, step[HelloStretchIdx.BASE_THETA]
            ):
                yield qi, ai
            for qi, ai in self.interpolate_xy(
                qi, xy0, dist, step[HelloStretchIdx.BASE_X]
            ):
                yield qi, ai
        else:
            theta = theta0
        # update angle
        if np.abs(thetag - theta) > theta_tol:
            for qi, ai in self.interpolate_angle(
                qi, theta, thetag, step[HelloStretchIdx.BASE_THETA]
            ):
                yield qi, ai
        # Finally interpolate the whole joint space
        for qi, ai in self.interpolate_arm(qi, qg, step):
            yield qi, ai

    def get_link_pose(self, link_name, q=None):
        if q is not None:
            self.set_config(q)
        return self.ref.get_link_pose(link_name)

    def fk(self, q=None, as_matrix=False):
        """forward kinematics"""
        pose = self.get_link_pose(self.ee_link_name, q)
        if as_matrix:
            return to_matrix(*pose)
        return pose

    def update_head(self, qi, look_at):
        qi[HelloStretchIdx.HEAD_PAN] = look_at[0]
        qi[HelloStretchIdx.HEAD_TILT] = look_at[1]
        return qi

    def update_gripper(self, qi, open=True):
        """update target state for gripper"""
        if open:
            qi[HelloStretchIdx.GRIPPER] = STRETCH_GRIPPER_OPEN
        else:
            qi[HelloStretchIdx.GRIPPER] = STRETCH_GRIPPER_CLOSE
        return qi

    def interpolate_xy(self, qi, xy0, dist, step=0.1):
        """just move forward with step to target distance"""
        # create a trajectory here
        x, y = xy0
        theta = qi[HelloStretchIdx.BASE_THETA]
        while dist > 0:
            qi = self.update_head(qi.copy(), self.look_front)
            ai = np.zeros(self.dof)
            if dist > step:
                dx = step
            else:
                dx = dist
            dist -= dx
            x += np.cos(theta) * dx
            y += np.sin(theta) * dx
            # x += np.sin(theta) * dx
            # y += np.cos(theta) * dx
            qi[HelloStretchIdx.BASE_X] = x
            qi[HelloStretchIdx.BASE_Y] = y
            ai[0] = dx
            yield qi, ai

    def _to_ik_format(self, q):
        qi = np.zeros(self.ik_solver.number_of_joints)
        qi[0] = q[HelloStretchIdx.BASE_X]
        qi[1] = q[HelloStretchIdx.BASE_Y]
        qi[2] = q[HelloStretchIdx.BASE_THETA]
        qi[3] = q[HelloStretchIdx.LIFT]
        # Next 4 are all arm joints
        arm_ext = q[HelloStretchIdx.ARM] / 4.0
        qi[4] = arm_ext
        qi[5] = arm_ext
        qi[6] = arm_ext
        qi[7] = arm_ext
        # Wrist joints
        qi[8] = q[HelloStretchIdx.WRIST_YAW]
        qi[9] = q[HelloStretchIdx.WRIST_PITCH]
        qi[10] = q[HelloStretchIdx.WRIST_ROLL]
        return qi

    def _to_plan_format(self, q):
        qi = np.zeros(self.dof)
        qi[HelloStretchIdx.BASE_X] = q[0]
        qi[HelloStretchIdx.BASE_Y] = q[1]
        qi[HelloStretchIdx.BASE_THETA] = q[2]
        qi[HelloStretchIdx.LIFT] = q[3]
        # Arm is sum of the next four joints
        qi[HelloStretchIdx.ARM] = q[4] + q[5] + q[6] + q[7]
        qi[HelloStretchIdx.WRIST_YAW] = q[8]
        qi[HelloStretchIdx.WRIST_PITCH] = q[9]
        qi[HelloStretchIdx.WRIST_ROLL] = q[10]
        return qi

    def ik(self, pose, q0):
        pos, rot = pose
        se3 = pb.getMatrixFromQuaternion(rot)
        pose = np.eye(4)
        pose[:3, :3] = np.array(se3).reshape(3, 3)
        x, y, z = pos
        pose[:3, 3] = np.array([x, y, z - self.base_height])
        q = self.ik_solver.ik(pose, self._to_ik_format(q0))
        if q is not None:
            return self._to_plan_format(q)
        else:
            return None

    def lift_arm_ik_from_matrix(self, pose_matrix, q0):
        end_arm_idx = 3 + self.lift_arm_ik_solver.number_of_joints
        q0 = self._to_ik_format(q0)
        ee_pose = self.get_ee_pose()
        print(ee_pose[0])
        print(ee_pose[1])
        print(pose_matrix[:3, 3])
        w, x, y, z = tra.quaternion_from_matrix(pose_matrix)
        print([x, y, z, w])
        print("----------------")

        q = self.lift_arm_ik_solver.ik(pose_matrix, q0[3:end_arm_idx])

        se3 = pb.getMatrixFromQuaternion(ee_pose[1])
        pose2 = np.eye(4)
        pose2[:3, :3] = np.array(se3).reshape(3, 3)
        pose2[:3, 3] = np.array(ee_pose[0])
        q2 = self.lift_arm_ik_solver.ik(pose2, q0[3:end_arm_idx])
        print(q)
        print(q2)

        if q is not None:
            res = q0.copy()
            res[3:end_arm_idx] = q
            res = self._to_plan_format(res)
            return res
        else:
            return None

    def static_ik(self, pose_query, q0):
        """constrainted ik, do not move the base?"""
        x0, y0 = q0[:2]
        pos, rot = pose_query
        se3 = pb.getMatrixFromQuaternion(rot)
        pose = np.eye(4)
        pose[:3, :3] = np.array(se3).reshape(3, 3)
        x, y, z = pos
        x, y = x - x0, y - y0

        fwd = self.fk(q0)

        # Rotate the pose around z axis - stretch correction
        # This is a hard-coded hack for the Stretch robot's base
        # We also need to correct for the current base rotation
        pose0 = pose
        pose[:3, 3] = np.array([x, y, z - self.base_height])
        R_correct = tra.euler_matrix(0, 0, np.pi / 2 - q0[2])
        # Now apply the rotation correction
        pose = R_correct.T @ pose

        # Assume base is stuck in place
        q0_zeroed = q0.copy()
        q0_zeroed[:3] = np.zeros(3)

        q = self.static_ik_solver.ik(pose, self._to_ik_format(q0_zeroed)[2:])
        if q is not None:
            res = np.zeros(len(q) + 2)
            res[2:] = q
            # Correct for base theta - since we zeroed this before we need to undo that
            res[2] = res[2] - q0[2]
            # Convert into the planner format
            res = self._to_plan_format(res)
            res[:2] = np.array([x0, y0])
            res[HelloStretchIdx.HEAD_PAN :] = q0[HelloStretchIdx.HEAD_PAN :]
            # TODO remove debug code
            # Update the config - useful for debugging
            # self.set_config(res)
            return res
        else:
            return None

    def get_ee_pose(self, q=None):
        if q is not None:
            self.set_config(q)
        return self.ref.get_link_pose(STRETCH_GRASP_FRAME)

    def update_look_front(self, q):
        """look in front so we can see the floor"""
        return self.update_head(q, self.look_front)

    def update_look_ahead(self, q):
        """look straight ahead; cannot see terrain"""
        return self.update_head(q, self.look_ahead)

    def update_look_at_ee(self, q):
        """turn and look at ee area"""
        return self.update_head(q, self.look_at_ee)

    def interpolate_angle(self, qi, theta0, thetag, step=0.1):
        """just rotate to target angle"""
        if theta0 > thetag:
            thetag2 = thetag + 2 * np.pi
        else:
            thetag2 = thetag - 2 * np.pi
        dist1 = np.abs(thetag - theta0)
        dist2 = np.abs(thetag2 - theta0)
        # TODO remove debug code
        # print("....", qi)
        print("interp from", theta0, "to", thetag, "or maybe", thetag2)
        # print("dists =", dist1, dist2)
        if dist2 < dist1:
            dist = dist2
            thetag = thetag2
        else:
            dist = dist1
        # Dumb check to see if we can get the rotation right
        # if dist > np.pi:
        #    thetag -= np.pi/2
        #    dist = np.abs(thetag - theta0)
        # TODO remove debug code
        # print(" > interp from", theta0, "to", thetag)
        # Getting the direction right here
        dirn = 1.0 if thetag > theta0 else -1.0
        while dist > 0:
            qi = qi.copy()
            ai = np.zeros(self.dof)
            # TODO: we should handle look differently
            # qi = self.update_head(qi, self.look_front)
            if dist > step:
                dtheta = step
            else:
                dtheta = dist
            dist -= dtheta
            ai[2] = dirn * dtheta
            qi[HelloStretchIdx.BASE_THETA] += dirn * dtheta
            yield qi, ai

    def interpolate_arm(self, q0, qg, step=None):
        if step is None:
            step = self.default_step
        qi = q0
        while np.any(np.abs(qi - qg) > self.default_tols):
            qi = qi.copy()
            ai = qi.copy()
            ai[:3] = np.zeros(3)  # action does not move the base
            # TODO: we should handle look differently
            qi = self.update_head(qi.copy(), self.look_at_ee)
            dq = qg - qi
            dq = np.clip(dq, -1 * step, step)
            qi += dq
            qi = self.update_head(qi, self.look_at_ee)
            yield qi, ai

    def is_colliding(self, other):
        return self.ref.is_colliding(other)

    def extend_arm_to(self, q, arm):
        """
        Extend the arm by a certain amound
        Move the base as well to compensate.

        This is purely a helper function to make sure that we can find poses at which we can
        extend the arm in order to grasp.
        """
        a0 = q[HelloStretchIdx.ARM]
        a1 = arm
        q = q.copy()
        q[HelloStretchIdx.ARM] = a1
        x = q[HelloStretchIdx.BASE_X]
        y = q[HelloStretchIdx.BASE_Y]
        theta = q[HelloStretchIdx.BASE_THETA] + np.pi / 2
        da = a1 - a0
        dx, dy = da * np.cos(theta), da * np.sin(theta)
        q[HelloStretchIdx.BASE_X] += dx
        q[HelloStretchIdx.BASE_Y] += dy
        return q

    def validate(self, q=None, ignored=[], distance=0.0, verbose=False):
        """
        Check collisions against different obstacles
        q = configuration to test
        ignored = other objects to NOT check against
        """
        self.set_config(q)
        # Check robot height
        if q[HelloStretchIdx.LIFT] >= 1.0:
            return False
        # Check links against obstacles
        for name, obj in self.backend.objects.items():
            if obj.id == self.ref.id:
                continue
            elif obj.id in ignored:
                continue
            elif self.ref.is_colliding(obj, distance=distance):
                if verbose:
                    print("colliding with", name)
                return False
        return True


if __name__ == "__main__":
    robot = HelloStretch()
    q0 = STRETCH_HOME_Q.copy()
    q1 = STRETCH_HOME_Q.copy()
    q0[2] = -1.18
    q1[2] = -1.1
    for state, action in robot.interpolate_angle(q0, q0[2], q1[2]):
        print(action)

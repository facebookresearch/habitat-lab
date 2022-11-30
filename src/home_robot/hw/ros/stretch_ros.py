import argparse
import numpy as np
import rospy
import ros_numpy
import threading
import time
import timeit
import sys
import trimesh.transformations as tra


from home_robot.hw.ros.abstract import AbstractStretchInterface
from home_robot.agent.motion.robot import HelloStretch, HelloStretchIdx
from home_robot.agent.motion.robot import STRETCH_HOME_Q
from home_robot.hw.ros.camera import RosCamera

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
import tf2_ros
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger


ROS_ARM_JOINTS = ["joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3"]
ROS_LIFT_JOINT = "joint_lift"
ROS_GRIPPER_FINGER = "joint_gripper_finger_left"
# ROS_GRIPPER_FINGER2 = "joint_gripper_finger_right"
ROS_HEAD_PAN = "joint_head_pan"
ROS_HEAD_TILT = "joint_head_tilt"
ROS_WRIST_YAW = "joint_wrist_yaw"
ROS_WRIST_PITCH = "joint_wrist_pitch"
ROS_WRIST_ROLL = "joint_wrist_roll"


ROS_TO_CONFIG = {
    ROS_LIFT_JOINT: HelloStretchIdx.LIFT,
    ROS_GRIPPER_FINGER: HelloStretchIdx.GRIPPER,
    # ROS_GRIPPER_FINGER2: HelloStretchIdx.GRIPPER,
    ROS_WRIST_YAW: HelloStretchIdx.WRIST_YAW,
    ROS_WRIST_PITCH: HelloStretchIdx.WRIST_PITCH,
    ROS_WRIST_ROLL: HelloStretchIdx.WRIST_ROLL,
    ROS_HEAD_PAN: HelloStretchIdx.HEAD_PAN,
    ROS_HEAD_TILT: HelloStretchIdx.HEAD_TILT,
}


CONFIG_TO_ROS = {}
for k, v in ROS_TO_CONFIG.items():
    if v not in CONFIG_TO_ROS:
        CONFIG_TO_ROS[v] = []
    CONFIG_TO_ROS[v].append(k)
CONFIG_TO_ROS[HelloStretchIdx.ARM] = ROS_ARM_JOINTS
# ROS_JOINT_NAMES += ROS_ARM_JOINTS


class HelloStretchROSInterface(AbstractStretchInterface):
    """Indicate which joints to use + how to control the robot"""

    exec_tol = np.array(
        [
            1e-3,
            1e-3,
            0.01,  # x y theta
            0.005,  # lift
            0.01,  # arm
            1.0,  # gripper - this never works
            # 0.015, 0.015, 0.015,  #wrist variables
            0.05,
            0.05,
            0.05,  # wrist variables
            0.1,
            0.1,  # head  and tilt
        ]
    )

    dist_tol = 1e-4
    theta_tol = 1e-3
    wait_time_step = 1e-3

    base_link = "base_link"
    odom_link = "map"

    def config_to_ros_msg(self, q):
        """convert into a joint state message"""
        msg = JointTrajectoryPoint()
        msg.positions = [0.0] * len(self.ros_joint_names)
        idx = 0
        for i in range(3, self.dof):
            names = CONFIG_TO_ROS[i]
            for _ in names:
                # Only for arm - but this is a dumb way to check
                if "arm" in names[0]:
                    msg.positions[idx] = q[i] / len(names)
                else:
                    msg.positions[idx] = q[i]
                idx += 1
        return msg

    def config_to_ros_trajectory_goal(self, q):
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = self.ros_joint_names
        trajectory_goal.trajectory.points = [self.config_to_ros_msg(q)]
        trajectory_goal.trajectory.header.stamp = rospy.Time.now()
        return trajectory_goal

    def config_from_ros_msg(self, msg):
        """convert from a joint state message"""
        q = np.zeros(self.dof)
        raise NotImplementedError()
        return q

    def get_pose(self, frame, lookup_time=None, timeout_s=None):
        """look up a particular frame in base coords"""
        if lookup_time is None:
            lookup_time = rospy.Time(0)  # return most recent transform
        if timeout_s is None:
            timeout_ros = rospy.Duration(0.1)
        else:
            timeout_ros = rospy.Duration(timeout_s)
        try:
            stamped_transform = self.tf2_buffer.lookup_transform(
                self.odom_link, frame, lookup_time, timeout_ros
            )
            pose_mat = ros_numpy.numpify(stamped_transform.transform)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            print("!!! Lookup failed from", self.base_link, "to", self.odom_link, "!!!")
            return None
        return pose_mat

    def get_base_pose(self, lookup_time=None, timeout_s=None):
        """lookup the base pose using TF2 buffer

        Based on this:
            https://github.com/hello-robot/stretch_ros/blob/master/hello_helpers/src/hello_helpers/hello_misc.py#L213
        """
        if lookup_time is None:
            lookup_time = rospy.Time(0)  # return most recent transform
        if timeout_s is None:
            timeout_ros = rospy.Duration(0.1)
        else:
            timeout_ros = rospy.Duration(timeout_s)
        try:
            # stamped_transform =  self.tf2_buffer.lookup_transform(self.base_link, self.odom_link,
            stamped_transform = self.tf2_buffer.lookup_transform(
                self.odom_link, self.base_link, lookup_time, timeout_ros
            )
            pose_mat = ros_numpy.numpify(stamped_transform.transform)

            r0 = pose_mat[:2, 3]
            _, _, theta = tra.euler_from_matrix(pose_mat)
            return (r0[0], r0[1], theta), stamped_transform.header.stamp
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            print("!!! Lookup failed from", self.base_link, "to", self.odom_link, "!!!")
            return None, None

    def __del__(self):
        """overwrite delete function since we do not need it any more"""
        pass

    def _js_cb(self, msg):
        # loop over all joint state info
        pos, vel, trq = np.zeros(self.dof), np.zeros(self.dof), np.zeros(self.dof)
        for name, p, v, e in zip(msg.name, msg.position, msg.velocity, msg.effort):
            # Check name etc
            if name in ROS_ARM_JOINTS:
                pos[HelloStretchIdx.ARM] += p
                vel[HelloStretchIdx.ARM] += v
                trq[HelloStretchIdx.ARM] += e
            elif name in ROS_TO_CONFIG:
                idx = ROS_TO_CONFIG[name]
                pos[idx] = p
                vel[idx] = v
                trq[idx] = e
        trq[HelloStretchIdx.ARM] /= 4
        with self._js_lock:
            self.pos, self.vel, self.frc = pos, vel, trq

    def update(self):
        # Return a full state for the robot
        pos, _ = self.get_base_pose()
        if pos is not None:
            x, y, theta = pos
        else:
            x, y, theta = 0.0, 0.0, 0.0
        with self._js_lock:
            pos, vel = self.pos.copy(), self.vel.copy()
        pos[:3] = np.array([x, y, theta])
        self.model.set_config(pos)
        return pos, vel

    def _get_linear_plan(self, q0, qg, max_plan_length=10):
        plan = []
        for q, a in self.model.interpolate(q0, qg):
            plan.append(a)
            # print("Q", q)
            # print("a", a)
        return plan

    def look_at(self, xyz):
        print("LOOKING AT", xyz)
        while not rospy.is_shutdown():
            q0, _ = self.update()
            plan = self.model.plan_look_at(q0, xyz)
            print(
                "base theta =", plan[0][2], "yaw =", plan[1][HelloStretchIdx.HEAD_TILT]
            )
            if np.abs(plan[0][2]) < 0.05:
                break
            self.goto_theta(plan[0][2])
            rospy.sleep(0.5)
        self.goto(plan[1], wait=True)

    def move_base(self, pos=None, theta=None, verbose=True, pos_tol=0.075, wait_t=1.0):
        # multi-step action. first rotate...
        # pos = (x, y) theta
        pos_reached = False
        q0 = None
        while not rospy.is_shutdown():
            # loop - update the pose we think we are in
            q0, _ = self.update()
            q1 = q0.copy()
            if pos is None:
                dist = 0
            else:
                dist = np.linalg.norm(q0[:2] - pos)
            if dist > pos_tol and not pos_reached:
                q1[:2] = pos
            else:
                # In order to stop us from getting caught in an i
                pos_reached = True
            # Try to get to the final position
            print("Distance to goal:", dist)
            print("Theta goal =", theta, "curr =", q0[2])
            if theta is not None:
                q1[2] = theta
            plan = self._get_linear_plan(q0, q1)
            if len(plan) < 1:
                break
            else:
                for i, action in enumerate(plan):
                    # print(action)
                    print(i, "move:", action[0], "rotate:", action[2])
                    if np.abs(action[0]) > 0.05 and not pos_reached:
                        if verbose:
                            print(i, "move x", action[0])
                        self.goto_x(action[0])
                        break
                    elif np.abs(action[2]) > 0.02:
                        if verbose:
                            print(i, "move theta", action[2])
                        self.goto_theta(action[2])
                        break
                    else:
                        # we're close enough the actions are tiny
                        if verbose:
                            print(i, "actions are tiny")
                else:
                    if verbose:
                        print("no remaining actions worth doing")
                    break
                rospy.sleep(wait_t)
        return q0

    def goto_x(self, x, wait=False, verbose=True):
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = [
            "translate_mobile_base",
        ]
        msg = JointTrajectoryPoint()
        msg.positions = [x]
        trajectory_goal.trajectory.points = [msg]
        trajectory_goal.trajectory.header.stamp = rospy.Time.now()
        self.trajectory_client.send_goal(trajectory_goal)
        if wait:
            #  Waiting for result seems to hang
            self.trajectory_client.wait_for_result()
            # self.wait(q, max_wait_t, True, verbose)
            print("-- TODO: wait for xy")
        return True

    def goto_theta(self, theta, wait=False, verbose=True):
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = [
            "rotate_mobile_base",
        ]
        msg = JointTrajectoryPoint()
        msg.positions = [theta]
        trajectory_goal.trajectory.points = [msg]
        trajectory_goal.trajectory.header.stamp = rospy.Time.now()
        self.trajectory_client.send_goal(trajectory_goal)
        if wait:
            self.trajectory_client.wait_for_result()
            # self.wait(q, max_wait_t, True, verbose)
            print("-- TODO: wait for theta")
        return True

    def _interp(self, x1, x2, num_steps=10):
        diff = x2 - x1
        rng = np.arange(num_steps + 1) / num_steps
        rng = rng[:, None].repeat(3, axis=1)
        diff = diff[None].repeat(num_steps + 1, axis=0)
        x1 = x1[None].repeat(num_steps + 1, axis=0)
        return x1 + (rng * diff)

    def goto_wrist(self, roll, pitch, yaw, verbose=False, wait=False):
        """Separate out wrist commands from everything else"""
        q, _ = self.update()
        r0, p0, y0 = (
            q[HelloStretchIdx.WRIST_ROLL],
            q[HelloStretchIdx.WRIST_PITCH],
            q[HelloStretchIdx.WRIST_YAW],
        )
        print("--------")
        print("roll", roll, "curr =", r0)
        print("pitch", pitch, "curr =", p0)
        print("yaw", yaw, "curr =", y0)
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = [
            ROS_WRIST_ROLL,
            ROS_WRIST_PITCH,
            ROS_WRIST_YAW,
        ]
        n_pts = 10
        # for i, q in enumerate(self._interp(np.array([r0, p0, y0]), np.array([roll, pitch, yaw]), n_pts)):
        #    pt = JointTrajectoryPoint()
        #    pt.positions = q
        #    pt.time_from_start = rospy.Time(i / n_pts)
        #    trajectory_goal.trajectory.points.append(pt)
        #    pt = JointTrajectoryPoint()
        #    pt.positions = q
        pt = JointTrajectoryPoint()
        pt.positions = [roll, pitch, yaw]
        trajectory_goal.trajectory.points = [pt]
        trajectory_goal.trajectory.header.stamp = rospy.Time.now()
        self.trajectory_client.send_goal(trajectory_goal)

    def goto(self, q, move_base=False, wait=True, max_wait_t=10.0, verbose=False):
        """some of these params are unsupported"""
        goal = self.config_to_ros_trajectory_goal(q)
        self.trajectory_client.send_goal(goal)
        if wait:
            #  Waiting for result seems to hang
            # self.trajectory_client.wait_for_result()
            print("waiting for result...")
            self.wait(q, max_wait_t, not move_base, verbose)
        return True

    def _mode_cb(self, msg):
        self.mode = msg.data

    def in_position_mode(self):
        return self.mode == "position"

    def get_images(self, filter_depth=True, compute_xyz=True):
        """helper logic to get images from the robot's camera feed"""
        rgb = self.rgb_cam.get()
        if filter_depth:
            dpt = self.dpt_cam.get_filtered()
        else:
            dpt = self.dpt_cam.get()
        if compute_xyz:
            xyz = self.dpt_cam.depth_to_xyz(self.dpt_cam.fix_depth(dpt))
            imgs = [rgb, dpt, xyz]
        else:
            imgs = [rgb, dpt]
            xyz = None

        # Get xyz in base coords for later
        imgs = [np.rot90(np.fliplr(np.flipud(x))) for x in imgs]

        if xyz is not None:
            xyz = imgs[-1]
            H, W = rgb.shape[:2]
            xyz = xyz.reshape(-1, 3)

            # Rotate the sretch camera so that top of image is "up"
            R_stretch_camera = tra.euler_matrix(0, 0, -np.pi / 2)[:3, :3]
            xyz = xyz @ R_stretch_camera
            xyz = xyz.reshape(H, W, 3)
            imgs[-1] = xyz

        return imgs

    def __init__(
        self,
        model=None,
        visualize_planner=False,
        root=".",
        init_cameras=True,
        depth_buffer_size=5,
    ):
        """Create an interface into ROS execution here. This one needs to connect to:
            - joint_states to read current position
            - tf for SLAM
            - FollowJointTrajectory for arm motions

        Based on this code:
        https://github.com/hello-robot/stretch_ros/blob/master/hello_helpers/src/hello_helpers/hello_misc.py
        """

        # No hardware interface here for the ROS code
        if model is None:
            model = HelloStretch(visualize=visualize_planner, root=root)
        self.model = model  # This is the model
        self.dof = model.dof

        if init_cameras:
            print("Creating cameras...")
            self.rgb_cam = RosCamera("/camera/color")
            self.dpt_cam = RosCamera(
                "/camera/aligned_depth_to_color", buffer_size=depth_buffer_size
            )
            print("Waiting for camera images...")
            self.rgb_cam.wait_for_image()
            self.dpt_cam.wait_for_image()
            print("..done.")
            print("rgb frame =", self.rgb_cam.get_frame())
            print("dpt frame =", self.dpt_cam.get_frame())
            # camera_pose = self.get_pose(self.rgb_cam.get_frame())
            # print("camera rgb pose:")
            # print(camera_pose)
        else:
            self.rgb_cam, self.dpt_cam = None, None

        # Store latest joint state message - lock for access
        self._js_lock = threading.Lock()
        self.mode = ""
        self._mode_pub = rospy.Subscriber("mode", String, self._mode_cb, queue_size=1)
        rospy.sleep(0.5)
        self.switch_to_position = rospy.ServiceProxy("switch_to_position_mode", Trigger)
        print("Wait for mode service...")
        self.switch_to_position.wait_for_service()
        print("... done.")
        if not self.in_position_mode():
            print("Switching to position mode...")
            print(self.switch_to_position())

        # ROS stuff
        self.trajectory_client = actionlib.SimpleActionClient(
            "/stretch_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)
        self.joint_state_subscriber = rospy.Subscriber(
            "stretch/joint_states", JointState, self._js_cb, queue_size=100
        )
        print("Waiting for trajectory server...")
        server_reached = self.trajectory_client.wait_for_server(
            timeout=rospy.Duration(30.0)
        )
        print("... connected.")
        if not server_reached:
            print("ERROR: Failed to connect to arm action server.")
            rospy.signal_shutdown(
                "Unable to connect to arm action server. Timeout exceeded."
            )
            sys.exit()

        self.ros_joint_names = []
        for i in range(3, self.dof):
            self.ros_joint_names += CONFIG_TO_ROS[i]
        self.reset_state()


if __name__ == "__main__":
    # Create the robot
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node("hello_stretch_ros_test")
    print("Create ROS interface")
    rob = HelloStretchROSInterface(visualize_planner=True, init_cameras=False)
    print("Wait...")
    rospy.sleep(0.5)  # Make sure we have time to get ROS messages
    for i in range(1):
        q = rob.update()
        print(rob.get_base_pose())
    print("--------------")
    print("We have updated the robot state. Now test goto.")

    home_q = STRETCH_HOME_Q
    model = rob.get_model()
    q = model.update_look_front(home_q.copy())
    rob.goto(q, move_base=False, wait=True)

    # Robot - look at the object because we are switching to grasping mode
    # Send robot to home_q + wait
    q = model.update_look_at_ee(home_q.copy())
    rob.goto(q, move_base=False, wait=True)
    # Raise the arm
    q[HelloStretchIdx.ARM] = 0.06
    q[HelloStretchIdx.LIFT] = 0.85
    rob.goto(q, move_base=False, wait=True)
    # Rotate the gripper out
    q[HelloStretchIdx.WRIST_YAW] = 0
    # Send it to pregrasp + wait
    q[HelloStretchIdx.ARM] = 0.4
    rob.goto(q, move_base=False, wait=True, verbose=False)
    time.sleep(0.5)
    qi, _ = rob.update()
    ee_pose = model.fk(qi)
    print("EE pose:", ee_pose[0])

    # do static ik here
    q0, _ = rob.update()
    fwd = model.fk(q0)
    print("forward kinematics")
    print(fwd)
    fwd[0][2] -= 0.1
    q1 = model.static_ik(fwd, q0)
    print(q0)
    print(q1)
    assert q1 is not None
    rob.goto(q1, move_base=False, wait=True)

    q[HelloStretchIdx.WRIST_YAW] = np.pi
    q[HelloStretchIdx.ARM] = 0.06
    rob.goto(q, move_base=False, wait=True, verbose=False)
    rob.goto(home_q, move_base=False, wait=True, verbose=False)

    """
    q0 = [-1.91967010e-01 -3.07846069e-02 -6.65076590e-01  5.00007841e-01
      5.99901370e-02  2.19929959e-01 -1.53398079e-03 -1.07378655e-02
      2.99957160e+00 -1.56995219e+00 -7.77738162e-01]
    qi = [-0.19196701 -0.03078461  0.46193263  0.98585885  0.3996423   0.
      3.06063741 -1.04642321  2.79234818 -1.56995219 -0.77773816]
  """

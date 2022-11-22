import rospy
import timeit
import numpy as np
import sophus as sp

from home_robot.hw.ros.stretch_ros import HelloStretchROSInterface
from home_robot.agent.motion.robot import STRETCH_HOME_Q, HelloStretchIdx
from home_robot.agent.perception.detectron2_segmentation import Detectron2Segmentation
from home_robot.agent.perception.constants import coco_categories
from home_robot.hw.ros.grasp_helper import GraspClient as RosGraspClient
from home_robot.utils.pose import to_pos_quat

from home_robot.client import LocalHelloRobot

import matplotlib.pyplot as plt

visualize_masks = False


def try_executing_grasp(rob, grasp) -> bool:
    """Try executing a grasp."""

    q, _ = rob.update()

    # Convert grasp pose to pos/quaternion
    grasp_pose = to_pos_quat(grasp)
    print("grasp xyz =", grasp_pose[0])

    # If can't plan to reach grasp, return
    qi = model.static_ik(grasp_pose, q)
    if qi is not None:
        model.set_config(qi)
    else:
        print(" --> ik failed")
        return False

    # Standoff 8 cm above grasp position
    q_standoff = qi.copy()
    # q_standoff[HelloStretchIdx.LIFT] += 0.08   # was 8cm, now more
    q_standoff[HelloStretchIdx.LIFT] += 0.1

    # Actual grasp position
    q_grasp = qi.copy()

    if q_standoff is not None:
        # If standoff position invalid, return
        if not model.validate(q_standoff):
            print("invalid standoff config:", q_standoff)
            return False
        print("found standoff")

        # Go to the grasp and try it
        q[HelloStretchIdx.LIFT] = 0.99
        rob.goto(q, move_base=False, wait=True, verbose=False)
        # input('--> go high')
        q_pre = q.copy()
        q_pre[5:] = q_standoff[5:]
        q_pre = model.update_gripper(q_pre, open=True)
        rob.move_base(theta=q_standoff[2])
        rospy.sleep(2.0)
        rob.goto(q_pre, move_base=False, wait=False, verbose=False)
        model.set_config(q_standoff)
        # input('--> gripper ready; go to standoff')
        q_standoff = model.update_gripper(q_standoff, open=True)
        rob.goto(q_standoff, move_base=False, wait=True, verbose=False)
        # input('--> go to grasp')
        rob.move_base(theta=q_grasp[2])
        rospy.sleep(2.0)
        rob.goto(q_pre, move_base=False, wait=False, verbose=False)
        model.set_config(q_grasp)
        q_grasp = model.update_gripper(q_grasp, open=True)
        rob.goto(q_grasp, move_base=False, wait=True, verbose=True)
        # input('--> close the gripper')
        q_grasp = model.update_gripper(q_grasp, open=False)
        rob.goto(q_grasp, move_base=False, wait=False, verbose=True)
        rospy.sleep(2.0)

        # Move back to standoff pose
        q_standoff = model.update_gripper(q_standoff, open=False)
        rob.goto(q_standoff, move_base=False, wait=True, verbose=False)

        # Move back to original pose
        q_pre = model.update_gripper(q_pre, open=False)
        rob.goto(q_pre, move_base=False, wait=True, verbose=False)

        # We completed the grasp
        return True


def divergence_from_vertical_grasp(grasp):
    dirn = grasp[:3, 2]
    theta_x = np.abs(np.arctan(dirn[0] / dirn[2]))
    theta_y = np.abs(np.arctan(dirn[1] / dirn[2]))
    return theta_x, theta_y


if __name__ == "__main__":
    # Create the robot
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node("hello_stretch_ros_test")
    print("Create ROS interface")
    rob = HelloStretchROSInterface(visualize_planner=False, init_cameras=False)
    print("Wait...")
    rospy.sleep(0.5)  # Make sure we have time to get ROS messages
    for i in range(1):
        q = rob.update()
        print(rob.get_base_pose())
    print("--------------")
    print("We have updated the robot state. Now test goto.")

    # Continuous nav interface
    nav_client = LocalHelloRobot(init_node=False)

    """
    rgb_cam = rob.rgb_cam
    dpt_cam = rob.dpt_cam
    rgb_cam.wait_for_image()
    dpt_cam.wait_for_image()
    """

    home_q = STRETCH_HOME_Q
    model = rob.get_model()
    q = model.update_look_front(home_q.copy())
    q = model.update_gripper(q, open=True)
    rob.goto(q, move_base=False, wait=True)

    # For video
    # Initial position
    # rob.move_base([0, 0], 0)
    # Move to before the chair
    # rob.move_base([0.5, -0.5], np.pi/2)
    print("Moving to grasp location...")
    xyt_goal = np.array([0.8, 0.4, np.pi / 2])
    # xyt_goal = np.array([0.5, 0., 0.])
    nav_client.set_nav_mode()
    nav_client.set_goal(xyt_goal)
    while True:
        xyt_robot = nav_client.get_base_state()
        if np.allclose(xyt_robot, xyt_goal, atol=0.15):
            break
        rospy.sleep(0.2)
    nav_client.set_pos_mode()

    rospy.sleep(0.5)  # Wait for robot to stabilize
    xyt_robot = nav_client.get_base_state()
    se3_robot = sp.SE3(
        sp.SO3.exp(np.array([0, 0, xyt_robot[2]])).matrix(),
        np.array([xyt_robot[0], xyt_robot[1], 0]),
    )

    q = model.update_look_at_ee(q)
    print("look at ee")
    rob.goto(q, wait=True)

    # Grasping
    x_vec = np.array([0.0, -0.5, 0.8])
    r_vec = np.array([-0.3, 2.5, -0.2])
    se3_grasp = sp.SE3(sp.SO3.exp(r_vec).matrix(), x_vec)
    se3_grasp_abs = se3_robot * se3_grasp
    grasp = se3_grasp_abs.matrix()
    print(f"Executing grasp: {grasp}")

    max_tries = 1
    for attempt in range(max_tries):
        print(f"=== Attempt {attempt+1} ===")
        theta_x, theta_y = divergence_from_vertical_grasp(grasp)
        print("with xy =", theta_x, theta_y)
        grasp_completed = try_executing_grasp(rob, grasp)
        if grasp_completed:
            print("Grasp successful")
            break

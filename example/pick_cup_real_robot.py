import rospy
import timeit
import numpy as np

from home_robot.hw.ros.stretch_ros import HelloStretchROSInterface
from home_robot.agent.motion.robot import STRETCH_HOME_Q, HelloStretchIdx
from home_robot.agent.perception.detectron2_segmentation import Detectron2Segmentation
from home_robot.agent.perception.constants import coco_categories
from home_robot.hw.ros.grasp_helper import GraspClient as RosGraspClient
from home_robot.utils.pose import to_pos_quat

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
        rob.goto(q, move_base=False, wait=True, verbose=False)  # move arm to top
        # input('--> go high')
        q_pre = q.copy()
        q_pre[5:] = q_standoff[5:]  # TODO: Add constants for joint indices
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
        rob.goto(q, move_base=False, wait=True, verbose=False)

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
    rob = HelloStretchROSInterface(visualize_planner=False)
    print("Wait...")
    rospy.sleep(0.5)  # Make sure we have time to get ROS messages
    for i in range(1):
        q = rob.update()
        print(rob.get_base_pose())
    print("--------------")
    print("We have updated the robot state. Now test goto.")

    rgb_cam = rob.rgb_cam
    dpt_cam = rob.dpt_cam
    rgb_cam.wait_for_image()
    dpt_cam.wait_for_image()

    grasp_client = RosGraspClient()
    segmentation_model = Detectron2Segmentation(
        sem_pred_prob_thr=0.9, sem_gpu_id=-1, visualize=True
    )

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

    q = model.update_look_at_ee(q)
    print("look at ee")
    rob.goto(q, wait=True)

    min_grasp_score = 0.0
    max_tries = 10
    min_obj_pts = 100
    for attempt in range(max_tries):
        rospy.sleep(1.0)

        t0 = timeit.default_timer()
        rgb, depth, xyz = rob.get_images()
        camera_pose = rob.get_pose(rgb_cam.get_frame())
        print("getting images + cam pose took", timeit.default_timer() - t0, "seconds")
        semantics, semantics_vis = segmentation_model.get_prediction(
            np.expand_dims(rgb[:, :, ::-1], 0), np.expand_dims(depth, 0)
        )
        cup_mask = semantics[0, :, :, coco_categories["cup"]]

        if visualize_masks:
            plt.figure()
            plt.subplot(131)
            plt.imshow(rgb)
            plt.subplot(132)
            plt.imshow(cup_mask)
            plt.subplot(133)
            _cup_mask = cup_mask[:, :, None]
            _cup_mask = np.repeat(_cup_mask, 3, axis=-1)
            plt.imshow(_cup_mask * rgb / 255.0)
            plt.show()

        num_cup_pts = np.sum(cup_mask)
        print("found this many cup points:", num_cup_pts)
        if num_cup_pts < min_obj_pts:
            continue

        mask_valid = depth > dpt_cam.near_val  # remove bad points
        mask_scene = mask_valid  # initial mask has to be good
        mask_scene = mask_scene.reshape(-1)

        predicted_grasps = grasp_client.request(
            xyz, rgb, cup_mask, frame=rgb_cam.get_frame()
        )
        if 0 not in predicted_grasps:
            print("no predicted grasps")
            continue
        predicted_grasps, scores = predicted_grasps[0]
        print("got this many grasps:", len(predicted_grasps))

        grasps = []
        for i, (score, grasp) in enumerate(zip(scores, predicted_grasps)):
            pose = grasp
            pose = camera_pose @ pose
            if score < min_grasp_score:
                continue

            # Get angles in world frame
            theta_x, theta_y = divergence_from_vertical_grasp(pose)
            theta = max(theta_x, theta_y)
            print(i, "score =", score, theta, "xy =", theta_x, theta_y)
            # Reject grasps that arent top down for now
            if theta > 0.3:
                continue
            grasps.append(pose)

        # Correct for the length of the Stretch gripper and the gripper upon
        # which Graspnet was trained
        grasp_offset = np.eye(4)
        grasp_offset[2, 3] = -0.09
        original_grasps = grasps.copy()
        for i, grasp in enumerate(grasps):
            grasps[i] = grasp @ grasp_offset

        for grasp in grasps:
            print("Executing grasp")
            print(grasp)
            theta_x, theta_y = divergence_from_vertical_grasp(grasp)
            print("with xy =", theta_x, theta_y)
            grasp_completed = try_executing_grasp(rob, grasp)
            if grasp_completed:
                break

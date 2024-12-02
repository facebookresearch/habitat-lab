import os
import pickle
import time

import cv2
import magnum as mn
import numpy as np
from geometry_utils import *
from sim_utils import *
from viz_utils import *

import habitat_sim
from habitat.tasks.rearrange.utils import IkHelper, is_pb_installed

ROBOT_FILE = (
    "/fsx-siro/jtruong/data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
)
DATA_FILE = "/fsx-siro/jtruong/repos/habitat-lab/sim2real/data/arm_pose.pkl"


def load_data(filepath):
    print("load data: ", filepath)
    with open(os.path.join(filepath), "rb") as handle:
        log_packet_list = pickle.load(handle)
    return log_packet_list


def simulate(sim, dt=1.0):
    r"""Runs physics simulation at 60FPS for a given duration (dt) optionally collecting and returning sensor observations."""
    target_time = sim.get_world_time() + dt
    while sim.get_world_time() < target_time:
        sim.step_physics(1.0 / 60.0)


def get_obs(sim, save_img=False):
    img = sim.get_sensor_observations()["rgb_camera"]
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if save_img:
        cur_time = int(time.time()) * 1000
        cv2.imwrite(
            f"/fsx-siro/jtruong/repos/habitat-lab/sim2real/output/tmp_{cur_time}.png",
            img_bgr,
        )
        time.sleep(1.0)
    return img_bgr


def test_position(sim, robot, iter=10):
    x = 0
    y = 0
    t = 0
    for i in range(iter):
        x += 0.2
        y += 0.1
        t += np.deg2rad(5)
        robot.set_base_position(x, y, t, relative=False)
        print("curr x, y yaw: ", robot.get_xy_yaw())
        get_obs(sim, save_img=True)


def test_rotation(sim, robot, iter=10):
    print("TEST ROTATION")
    axes = np.eye(3)
    names = ["roll pitch yaw".split(), "xyz"]
    values = [
        np.linspace(np.radians(-60), np.radians(60), 300),
        np.linspace(-1.5, 1.5, 150),
    ]
    imgs = []
    ctr = 0
    for idx in range(2):
        # for idx in range(0):
        for axis_idx in range(3):
            for val in values[idx]:
                global_T_base_std = (
                    mn.Matrix4.rotation(
                        mn.Rad(val), mn.Vector3(axes[axis_idx])
                    )
                    if idx == 0
                    else mn.Matrix4.translation(axes[axis_idx] * val)
                )
                global_T_base_std.translation = mn.Vector3(0, 0, 1.0)
                set_robot_base_transform(robot, global_T_base_std)
                base_rpy = np.degrees(
                    extract_roll_pitch_yaw(
                        get_robot_base_transform(robot).rotation()
                    )
                )
                ee_rpy = np.degrees(
                    extract_roll_pitch_yaw(get_ee_transform(robot).rotation())
                )
                img_bgr = get_obs(sim, save_img=False)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_rgb = add_text_to_image(
                    img_rgb,
                    f"base_rpy: {base_rpy[0]:.2f}, {base_rpy[1]:.2f}, {base_rpy[2]:.2f} \n ee_rpy: {ee_rpy[0]:.2f}, {ee_rpy[1]:.2f}, {ee_rpy[2]:.2f}",
                    top=True,
                )

                obs = {"color_sensor": img_rgb}
                imgs.append(obs)
                ctr += 1
    from habitat_sim.utils import viz_utils as vut

    print("len: ", len(imgs))
    vut.make_video(
        imgs,
        "color_sensor",
        "color",
        f"/fsx-siro/jtruong/repos/habitat-lab/sim2real/output/tmp_vid_{int(time.time())*1000}",
        open_vid=False,
    )
    return


def test_spot_rotation(sim, robot, iter=10):
    print("TEST SPOT ROTATION")
    axes = np.eye(3)
    names = ["roll pitch yaw".split(), "xyz"]
    values = [
        np.linspace(np.radians(-60), np.radians(60), 300),
        np.linspace(-1.5, 1.5, 150),
    ]
    imgs = []
    ctr = 0
    for idx in range(2):
        # for idx in range(0):
        for axis_idx in range(3):
            for val in values[idx]:
                global_T_base_std = (
                    mn.Matrix4.rotation(
                        mn.Rad(val), mn.Vector3(axes[axis_idx])
                    )
                    if idx == 0
                    else mn.Matrix4.translation(axes[axis_idx] * val)
                )
                global_T_base_std.translation = mn.Vector3(0, 0, 1.0)
                robot.set_robot_base_transform(global_T_base_std)
                base_rpy = np.degrees(
                    extract_roll_pitch_yaw(
                        robot.base_transformation.rotation()
                    )
                )
                ee_rpy = np.degrees(
                    extract_roll_pitch_yaw(robot.ee_transform(7).rotation())
                )
                img_bgr = get_obs(sim, save_img=False)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_rgb = add_text_to_image(
                    img_rgb,
                    f"base_rpy: {base_rpy[0]:.2f}, {base_rpy[1]:.2f}, {base_rpy[2]:.2f} \n ee_rpy: {ee_rpy[0]:.2f}, {ee_rpy[1]:.2f}, {ee_rpy[2]:.2f}",
                    top=True,
                )

                obs = {"color_sensor": img_rgb}
                imgs.append(obs)
                ctr += 1
    from habitat_sim.utils import viz_utils as vut

    print("len: ", len(imgs))
    vut.make_video(
        imgs,
        "color_sensor",
        "color",
        f"/fsx-siro/jtruong/repos/habitat-lab/sim2real/output/tmp_vid_{int(time.time())*1000}",
        open_vid=False,
    )
    return


def test_real_replay(sim, robot, use_joints=True):
    real_data = load_data(DATA_FILE)

    if not use_joints and is_pb_installed:
        arm_urdf = "/opt/hpcaas/.mounts/fs-03ee9f8c6dddfba21/jtruong/data/versioned_data/hab_spot_arm/urdf/spot_onlyarm.urdf"
        ik_helper = IkHelper(
            arm_urdf,
            np.deg2rad(np.array([0.0, -180, 0.0, 180.0, 0.0, 0.0, 0.0])),
        )

    for step_data in real_data:
        draw_axes(sim)
        x, y, t = step_data["base_pose_xyt"]
        global_T_base_std = mn.Matrix4.rotation(mn.Rad(t), mn.Vector3(0, 0, 1))
        global_T_base_std.translation = mn.Vector3(x, y, 0.0)
        robot.set_base_position(x, y, t)

        sh0, sh1, el0, el1, wr0, wr1 = step_data["arm_pose"]

        if "ee_pose" in step_data.keys():
            ee_xyz, ee_rpy = step_data["ee_pose"]

        if use_joints:
            robot.set_arm_joint_positions(
                np.array([sh0, sh1, 0.0, el0, el1, wr0, wr1])
            )
        else:
            joint_pos = np.array(robot.arm_joint_pos)
            joint_vel = np.zeros(joint_pos.shape)
            ik_helper.set_arm_state(joint_pos, joint_vel)

            arm_joints = ik_helper.calc_ik(ee_xyz, ee_rpy)

            robot.set_arm_joint_positions(arm_joints)
        # base_rpy = extract_roll_pitch_yaw(robot.base_transformation.rotation())
        # ee_rpy = extract_roll_pitch_yaw(robot.ee_transform().rotation())
        # print("base_rpy: ", base_rpy, "ee_rpy: ", ee_rpy)

        base_T_ee_pos, base_T_ee_quat = robot.get_ee_local_pose()
        print(
            "base_T_ee pose: ",
            base_T_ee_pos,
            ee_xyz,
            np.allclose(base_T_ee_pos, ee_xyz, atol=1e-1),
        )
        get_obs(sim, save_img=True)


def reset_robot(robot):
    start_tform = mn.Matrix4.translation(mn.Vector3(0, 0, 1))
    robot.set_robot_base_transform(start_tform)
    robot.leg_joint_pos = [0.0, 0.7, -1.5] * 4
    robot.set_arm_joint_positions(
        np.deg2rad(np.array([0.0, -180, 0.0, 180.0, 0.0, 0.0, 0.0]))
    )
    robot.close_gripper()


def main(sim):
    # visualize_axes(sim)
    make_ground_cube(sim)

    camera = sim.initialize_agent(agent_id=0)
    global_T_camera_hab = mn.Matrix4().look_at(
        eye=mn.Vector3(-1.0, 1.0, -1.0) * 3,
        target=mn.Vector3(0, 0, 0),
        up=mn.Vector3(0, 1, 0),
    )

    camera.set_state(magnum_to_agent_state(global_T_camera_hab))

    # robot = load_robot(sim, ROBOT_FILE)
    robot = load_spot_robot(sim, ROBOT_FILE)
    pt = np.array([-7.03365, 0.95533, -7.7762])
    mn_vec3 = mn.Vector3(*pt)
    mn_mat4 = mn.Matrix4.translation(mn_vec3)

    print("1: ", convert_conventions(mn_vec3))
    print("2: ", convert_conventions(mn_mat4).translation)

    # test_rotation(sim, robot.sim_obj)

    # reset_robot(robot)

    # test_real_replay(sim, robot)
    # test_position(sim, robot)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    with habitat_sim.Simulator(make_configuration()) as sim:
        main(sim)

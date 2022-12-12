#!/usr/bin/env python

"""
Copied from stretch-body/stretch_xbox_controller_teleop.py
"""
# TODO: actual link to original
# TODO: move this somewhere better, since it's in the "ros" folder but isn't actually ROS

from __future__ import print_function
import stretch_body.xbox_controller as xc

# import stretch_body.robot as rb
from stretch_body.hello_utils import *
import os
import time
import argparse

print_stretch_re_use()

"""parser = argparse.ArgumentParser(description=
                                 'Jog the robot from an XBox Controller  \n' +
                                 '-------------------------------------\n' +
                                 'Left Stick X:\t Rotate base \n' +
                                 'Left Stick Y:\t Translate base \n' +
                                 'Right Trigger:\t Fast base motion \n' +
                                 'Right Stick X:\t Translate arm \n' +
                                 'Right Stick Y:\t Translate lift \n' +
                                 'Left Button:\t Rotate wrist CCW \n' +
                                 'Right Button:\t Rotate wrist CW \n' +
                                 'A/B Buttons:\t Close/Open gripper \n' +
                                 'Left/Right Pad:\t Head Pan \n' +
                                 'Top/Bottom Pad:\t Head tilt \n' +
                                 'Y Button :\t Go to stow position \n ' +
                                 'Start Button:\t Home robot \n ' +
                                 'Back Button (2 sec):\t Shutdown computer \n ' +
                                 '-------------------------------------\n',
                                 formatter_class=argparse.RawTextHelpFormatter)

args = parser.parse_args()"""


class CommandToLinearMotion:
    def __init__(
        self, command_dead_zone_value, move_duration_s, max_distance_m, accel_m
    ):
        # This expects
        # a command value with a magnitude between 0.0 and 1.0, inclusive
        # a dead_zone with a magnitude greater than or equal to 0.0 and less than 1.0

        self.dead_zone = abs(command_dead_zone_value)
        self.move_duration_s = abs(move_duration_s)
        self.max_distance_m = abs(max_distance_m)
        self.accel_m = abs(accel_m)

        # check that the values are reasonable
        assert self.dead_zone >= 0.0
        assert (
            self.dead_zone <= 0.9
        ), "WARNING: CommandToLinearMotion.__init__ command_dead_zone_value is strangely large command_dead_zone_value = abs({0}) > 0.9.".format(
            command_dead_zone_value
        )
        assert (
            self.move_duration_s > 0.01
        ), "WARNING: CommandToLinearMotion.__init__ move_duration_s = abs({0}) <= 0.01 seconds, which is a short time for a single move.".format(
            move_duration_s
        )
        assert (
            self.move_duration_s <= 1.0
        ), "WARNING: CommandToLinearMotion.__init__ move_duration_s = abs({0}) > 1.0 seconds, which is a long time for a single move.".format(
            move_duration_s
        )
        assert (
            self.max_distance_m <= 0.3
        ), "WARNING: CommandToLinearMotion.__init__ max_distance_m = abs({0}) > 0.3 meters, which is a long distance for a single move.".format(
            max_distance_m
        )
        assert (
            self.accel_m <= 30.0
        ), "WARNING: CommandToLinearMotion.__init__ accel_m = abs({0}) > 30.0 m/s^2, which is very high (> 3 g).".format(
            accel_m
        )

    def get_dist_vel_accel(self, output_sign, command_value):
        # Larger commands attempt to move over larger distances in the
        # same amount of time by moving at higher velocities.
        c_val = abs(command_value)

        assert (
            c_val <= 1.0
        ), "ERROR: CommandToLinearMotion.get_dist_vel_accel given command value > 1.0, command_value = {0}".format(
            command_value
        )
        assert (
            c_val > self.dead_zone
        ), "ERROR: CommandToLinearMotion.get_dist_vel_accel the command should not be executed due to its value being within the dead zone: abs(command_value) = abs({0}) <= {1} = self.dead_zone".format(
            command_value, self.dead_zone
        )
        if 1:
            scale = (c_val - self.dead_zone) / (1.0 - self.dead_zone) ** 2
        else:
            scale = c_val - self.dead_zone
        d_m = scale * (self.max_distance_m / (1.0 - self.dead_zone))
        d_m = math.copysign(d_m, output_sign)
        v_m = (
            d_m / self.move_duration_s
        )  # average m/s for a move of distance d_m to last for time move_s
        a_m = self.accel_m
        return d_m, v_m, a_m


class CommandToRotaryMotion:
    def __init__(
        self, command_dead_zone_value, move_duration_s, max_angle_rad, accel_rad
    ):
        # This expects
        # a command value with a magnitude between 0.0 and 1.0, inclusive
        # a dead_zone with a magnitude greater than or equal to 0.0 and less than 1.0

        self.dead_zone = abs(command_dead_zone_value)
        self.move_duration_s = abs(move_duration_s)
        self.max_angle_rad = abs(max_angle_rad)
        self.accel_rad = abs(accel_rad)

        # check that the values are reasonable
        assert self.dead_zone >= 0.0
        assert (
            self.dead_zone <= 0.9
        ), "WARNING: CommandToRotaryMotion.__init__ command_dead_zone_value is strangely large command_dead_zone_value = abs({0}) > 0.9.".format(
            command_dead_zone_value
        )
        assert (
            self.move_duration_s > 0.01
        ), "WARNING: CommandToRotaryMotion.__init__ move_duration_s = abs({0}) <= 0.01 second, which is a short time for a single move.".format(
            move_duration_s
        )
        assert (
            self.move_duration_s <= 1.0
        ), "WARNING: CommandToRotaryMotion.__init__ move_duration_s = abs({0}) > 1.0 second, which is a long time for a single move.".format(
            move_duration_s
        )
        assert (
            self.max_angle_rad <= 0.7
        ), "WARNING: CommandToRotaryMotion.__init__ max_angle_rad = abs({0}) > 0.7 , which is a large angle for a single move (~40.0 deg).".format(
            max_angle_rad
        )
        assert (
            self.accel_rad <= 4.0 * 10
        ), "WARNING: CommandToRotaryMotion.__init__ accel_rad = abs({0}) > 4.0 rad/s^2, which is high.".format(
            accel_rad
        )

    def get_dist_vel_accel(self, output_sign, command_value):
        # Larger commands attempt to move over larger distances in the
        # same amount of time by moving at higher velocities.
        c_val = abs(command_value)
        assert (
            c_val <= 1.0
        ), "ERROR: CommandToRotaryMotion.get_dist_vel_accel given command value > 1.0, command_value = {0}".format(
            command_value
        )
        assert (
            c_val > self.dead_zone
        ), "ERROR: CommandToRotaryMotion.get_dist_vel_accel the command should not be executed due to its value being within the dead zone: abs(command_value) = abs({0}) <= {1} = self.dead_zone".format(
            command_value, self.dead_zone
        )

        scale = c_val - self.dead_zone
        d_r = scale * (self.max_angle_rad / (1.0 - self.dead_zone))
        d_r = math.copysign(d_r, output_sign)
        v_r = (
            d_r / self.move_duration_s
        )  # average m/s for a move of distance d_m to last for time move_s
        a_r = self.accel_rad
        return d_r, v_r, a_r


# ######################### HEAD ########################################
head_pan_target = 0.0
head_tilt_target = 0.0


def manage_head(robot, controller_state):
    global head_pan_target, head_tilt_target
    if not use_head_mapping:
        return
    head_pan_rad = deg_to_rad(40.0)  # 25.0 #40.0
    head_pan_accel = 14.0  # 15.0
    head_pan_vel = 7.0  # 6.0
    head_pan_slew_down = 0.15
    head_pan_slew_up = 0.15

    camera_pan_right = controller_state["right_pad_pressed"]
    camera_pan_left = controller_state["left_pad_pressed"]
    camera_tilt_up = controller_state["top_pad_pressed"]
    camera_tilt_down = controller_state["bottom_pad_pressed"]

    # Slew target to zero if no buttons pushed
    if not camera_pan_left and not camera_pan_right:
        if head_pan_target >= 0:
            head_pan_target = max(0, head_pan_target - head_pan_slew_down)
        else:
            head_pan_target = min(0, head_pan_target + head_pan_slew_down)
    # Or slew up to max
    if camera_pan_right:
        head_pan_target = max(head_pan_target - head_pan_slew_up, -head_pan_rad)
    if camera_pan_left:
        head_pan_target = min(head_pan_target + head_pan_slew_up, head_pan_rad)

    head_pan_command = (head_pan_target, head_pan_vel, head_pan_accel)
    if robot is not None:
        robot.head.move_by("head_pan", *head_pan_command)

    # Head tilt
    head_tilt_rad = deg_to_rad(40.0)  # 25.0 #40.0
    head_tilt_accel = 14.0  # 12.0  # 15.0
    head_tilt_vel = 7.0  # 10.0  # 6.0
    head_tilt_slew_down = 0.15
    head_tilt_slew_up = 0.15
    # Slew target to zero if no buttons pushed
    if not camera_tilt_up and not camera_tilt_down:
        if head_tilt_target >= 0:
            head_tilt_target = max(0, head_tilt_target - head_tilt_slew_down)
        else:
            head_tilt_target = min(0, head_tilt_target + head_tilt_slew_down)
    # Or slew up to max
    if camera_tilt_down:
        head_tilt_target = max(head_tilt_target - head_tilt_slew_up, -head_tilt_rad)
    if camera_tilt_up:
        head_tilt_target = min(head_tilt_target + head_tilt_slew_up, head_tilt_rad)

    head_tilt_command = (head_tilt_target, head_tilt_vel, head_tilt_accel)
    if robot is not None:
        robot.head.move_by("head_tilt", *head_tilt_command)

    return head_pan_command, head_tilt_command


# ######################### BASE ########################################
############################
# Regular Motion
dead_zone = 0.01  # 0.25 #0.1 #0.2 #0.3 #0.4
move_s = 0.6
max_dist_m = 0.06  # 0.04 #0.05
accel_m = 0.2  # 0.1
command_to_linear_motion = CommandToLinearMotion(dead_zone, move_s, max_dist_m, accel_m)

move_s = 0.05
max_dist_rad = 0.10  # 0.2 #0.25 #0.1 #0.09
accel_rad = 0.8  # 0.05
command_to_rotary_motion = CommandToRotaryMotion(
    dead_zone, move_s, max_dist_rad, accel_rad
)
############################
# Fast Motion
fast_move_s = 0.6
fast_max_dist_m = 0.12
fast_accel_m = 0.8
# fast, but unstable on thresholds: 0.6 s, 0.15 m, 0.8 m/s^2

fast_command_to_linear_motion = CommandToLinearMotion(
    dead_zone, fast_move_s, fast_max_dist_m, fast_accel_m
)
fast_move_s = 0.2
fast_max_dist_rad = 0.6
fast_accel_rad = 0.8
fast_command_to_rotary_motion = CommandToRotaryMotion(
    dead_zone, fast_move_s, fast_max_dist_rad, fast_accel_rad
)


############################


def manage_base(robot, controller_state):
    forward_command = controller_state["left_stick_y"]
    turn_command = controller_state["left_stick_x"]

    fast_navigation_mode = False
    navigation_mode_trigger = controller_state["right_trigger_pulled"]
    if navigation_mode_trigger > 0.5:
        fast_navigation_mode = True

    ##################
    # convert robot commands to robot movement
    # only allow a pure translation or a pure rotation command
    translation_command = None
    rotation_command = None

    if abs(forward_command) > abs(turn_command):
        if abs(forward_command) > dead_zone:
            output_sign = math.copysign(1, forward_command)
            if not fast_navigation_mode:
                d_m, v_m, a_m = command_to_linear_motion.get_dist_vel_accel(
                    output_sign, forward_command
                )
            else:
                d_m, v_m, a_m = fast_command_to_linear_motion.get_dist_vel_accel(
                    output_sign, forward_command
                )

            translation_command = (d_m, v_m, a_m)
            if robot is not None:
                robot.base.translate_by(d_m, v_m, a_m)
    else:
        if abs(turn_command) > dead_zone:
            output_sign = -math.copysign(1, turn_command)
            if not fast_navigation_mode:
                d_rad, v_rad, a_rad = command_to_rotary_motion.get_dist_vel_accel(
                    output_sign, turn_command
                )
            else:
                d_rad, v_rad, a_rad = fast_command_to_rotary_motion.get_dist_vel_accel(
                    output_sign, turn_command
                )

            rotation_command = (d_rad, v_rad, a_rad)
            if robot is not None:
                robot.base.rotate_by(d_rad, v_rad, a_rad)

    return translation_command, rotation_command


# ######################### LIFT & ARM  ########################################


def manage_lift_arm(robot, controller_state):
    lift_command = controller_state["right_stick_y"]
    arm_command = controller_state["right_stick_x"]

    converted_lift_command = None
    converted_arm_command = None

    if abs(lift_command) > dead_zone:
        output_sign = math.copysign(1, lift_command)
        d_m, v_m, a_m = command_to_linear_motion.get_dist_vel_accel(
            output_sign, lift_command
        )
        converted_lift_command = (d_m, v_m, a_m)

        if robot is not None:
            robot.lift.move_by(d_m, v_m, a_m)

    if abs(arm_command) > dead_zone:
        output_sign = math.copysign(1, arm_command)
        d_m, v_m, a_m = command_to_linear_motion.get_dist_vel_accel(
            output_sign, arm_command
        )
        converted_arm_command = (d_m, v_m, a_m)

        if robot is not None:
            robot.arm.move_by(d_m, v_m, a_m)

    return converted_lift_command, converted_arm_command


# ######################### END OF ARM  ########################################
wrist_yaw_target = 0.0
wrist_roll_target = 0.0
wrist_pitch_target = 0.0


def manage_end_of_arm(robot, controller_state):
    global wrist_yaw_target, wrist_roll_target, wrist_pitch_target
    wrist_yaw_left = controller_state["left_shoulder_button_pressed"]
    wrist_yaw_right = controller_state["right_shoulder_button_pressed"]

    close_gripper = controller_state["bottom_button_pressed"]
    open_gripper = controller_state["right_button_pressed"]

    wrist_yaw_rotate_rad = deg_to_rad(25.0)  # 60.0 #10.0 #5.0
    wrist_yaw_accel = 15.0  # 25.0 #30.0 #15.0 #8.0 #15.0
    wrist_yaw_vel = 25.0  # 10.0 #6.0 #3.0 #6.0
    wrist_yaw_slew_down = 0.15
    wrist_yaw_slew_up = 0.02

    wrist_roll_rotate_rad = deg_to_rad(25.0)  # 60.0 #10.0 #5.0
    wrist_roll_accel = 15.0  # 25.0 #30.0 #15.0 #8.0 #15.0
    wrist_roll_vel = 25.0  # 10.0 #6.0 #3.0 #6.0
    wrist_roll_slew_down = 0.15
    wrist_roll_slew_up = 0.02

    wrist_pitch_rotate_rad = deg_to_rad(25.0)  # 60.0 #10.0 #5.0
    wrist_pitch_accel = 25.0  # 25.0 #30.0 #15.0 #8.0 #15.0
    wrist_pitch_vel = 25.0  # 10.0 #6.0 #3.0 #6.0
    wrist_pitch_slew_down = 0.15
    wrist_pitch_slew_up = 0.02

    wrist_roll_command = None
    wrist_pitch_command = None
    gripper_command = None

    # Slew target to zero if no buttons pushed
    if not wrist_yaw_left and not wrist_yaw_right:
        if wrist_yaw_target >= 0:
            wrist_yaw_target = max(0, wrist_yaw_target - wrist_yaw_slew_down)
        else:
            wrist_yaw_target = min(0, wrist_yaw_target + wrist_yaw_slew_down)
    # Or slew up to max
    if wrist_yaw_left:
        wrist_yaw_target = min(
            wrist_yaw_target + wrist_yaw_slew_up, wrist_yaw_rotate_rad
        )
    if wrist_yaw_right:
        wrist_yaw_target = max(
            wrist_yaw_target - wrist_yaw_slew_up, -wrist_yaw_rotate_rad
        )

    wrist_yaw_command = (wrist_yaw_target, wrist_yaw_vel, wrist_yaw_accel)
    if robot is not None:
        robot.end_of_arm.move_by("wrist_yaw", *wrist_yaw_command)

    if use_dex_wrist_mapping:
        wrist_roll_cw = controller_state["right_pad_pressed"]
        wrist_roll_ccw = controller_state["left_pad_pressed"]
        wrist_pitch_down = controller_state["top_pad_pressed"]
        wrist_pitch_up = controller_state["bottom_pad_pressed"]

        if not wrist_roll_cw and not wrist_roll_ccw:
            if wrist_roll_target >= 0:
                wrist_roll_target = max(0, wrist_roll_target - wrist_roll_slew_down)
            else:
                wrist_roll_target = min(0, wrist_roll_target + wrist_roll_slew_down)
        if wrist_roll_cw:
            wrist_roll_target = max(
                wrist_roll_target - wrist_roll_slew_up, -wrist_roll_rotate_rad
            )
        if wrist_roll_ccw:
            wrist_roll_target = min(
                wrist_roll_target + wrist_roll_slew_up, wrist_roll_rotate_rad
            )

        if not wrist_pitch_up and not wrist_pitch_down:
            if wrist_pitch_target >= 0:
                wrist_pitch_target = max(0, wrist_pitch_target - wrist_pitch_slew_down)
            else:
                wrist_pitch_target = min(0, wrist_pitch_target + wrist_pitch_slew_down)
        if wrist_pitch_up:
            wrist_pitch_target = max(
                wrist_pitch_target - wrist_pitch_slew_up, -wrist_pitch_rotate_rad
            )
        if wrist_pitch_down:
            wrist_pitch_target = min(
                wrist_pitch_target + wrist_pitch_slew_up, wrist_pitch_rotate_rad
            )

        wrist_roll_command = (wrist_roll_target, wrist_pitch_vel, wrist_pitch_accel)
        wrist_pitch_command = (wrist_pitch_target, wrist_pitch_vel, wrist_pitch_accel)
        if robot is not None:
            robot.end_of_arm.move_by("wrist_roll", *wrist_roll_command)
            # print('WW',rad_to_deg(wrist_roll_target),robot.end_of_arm.motors['wrist_roll'].status['pos_ticks'])
            robot.end_of_arm.move_by("wrist_pitch", *wrist_pitch_command)

    if use_stretch_gripper_mapping:
        gripper_rotate_pct = 0.05  # TODO spowers: originally 10.0
        gripper_accel = None  # TODO spowers
        gripper_vel = None  # TODO spowers

        if robot is not None:
            gripper_accel = robot.end_of_arm.motors["stretch_gripper"].params["motion"][
                "max"
            ]["accel"]
            gripper_vel = robot.end_of_arm.motors["stretch_gripper"].params["motion"][
                "max"
            ]["vel"]

        if open_gripper:
            gripper_command = (gripper_rotate_pct, gripper_vel, gripper_accel)
        elif close_gripper:
            gripper_command = (-gripper_rotate_pct, gripper_vel, gripper_accel)

        if robot is not None and gripper_command is not None:
            robot.end_of_arm.move_by("stretch_gripper", *gripper_command)
    return wrist_yaw_command, wrist_roll_command, wrist_pitch_command, gripper_command


# ######################### SHUTDOWN  ########################################
shutdown_pc = False
ts_shutdown_start = 0


def manage_shutdown(robot, controller_state):
    global shutdown_pc, ts_shutdown_start
    if controller_state["select_button_pressed"]:
        if not shutdown_pc:
            ts_shutdown_start = time.time()
            shutdown_pc = True
        if time.time() - ts_shutdown_start > 2.0:
            robot.pimu.trigger_beep()
            robot.stow()
            robot.stop()
            time.sleep(1.0)
            os.system(
                "paplay --device=alsa_output.pci-0000_00_1f.3.analog-stereo /usr/share/sounds/ubuntu/stereo/desktop-logout.ogg"
            )
            os.system(
                "sudo shutdown now"
            )  # sudoers should be set up to not need a password
    else:
        shutdown_pc = False


# ######################### STOW and CALIBRATION  ########################################
def manage_stow(robot, controller_state):
    stow_robot = controller_state["top_button_pressed"]
    if stow_robot and robot.is_calibrated():
        # Reset motion params as fast for xbox
        v = robot.end_of_arm.motors["wrist_yaw"].params["motion"]["default"]["vel"]
        a = robot.end_of_arm.motors["wrist_yaw"].params["motion"]["default"]["accel"]
        robot.end_of_arm.motors["wrist_yaw"].set_motion_params(v, a)
        robot.stow()


first_home_warn = True


def manage_calibration(robot, controller_state):
    global first_home_warn
    calibrate_the_robot = controller_state["start_button_pressed"]
    if calibrate_the_robot:
        print("begin calibrating the robot")
        robot.home()
        print("finished calibrating the robot")
    else:
        if first_home_warn:
            print("press the start button to calibrate the robot")
        else:
            first_home_warn = False


########################### Check and wait for USB Devices ###########################


def check_usb_devices(wait_timeout=5):
    hello_devices = [
        "hello-wacc",
        "hello-motor-left-wheel",
        "hello-pimu",
        "hello-dynamixel-head",
        "hello-dynamixel-wrist",
        "hello-motor-arm",
        "hello-motor-right-wheel",
        "hello-motor-lift",
    ]

    print("Waiting for all the hello* devices ...")
    all_found = True
    for dev in hello_devices:
        if not wait_till_usb(dev, wait_timeout):
            all_found = False
    if all_found:
        print("Found all hello* devices.")
    return all_found


def wait_till_usb(usb, wait_timeout):
    s_ts = time.time()
    while time.time() - s_ts <= wait_timeout:
        devices = os.listdir("/dev")
        hello_devs = [dev for dev in devices if "hello" in dev]
        if usb in hello_devs:
            return True
    print("{} device not found.".format(usb))
    return False


# ######################### MAIN ########################################
use_head_mapping = True
use_dex_wrist_mapping = False
use_stretch_gripper_mapping = True


def set_use_dex_wrist_mapping(new_val):
    # TODO spowers: hacky
    global use_dex_wrist_mapping
    use_dex_wrist_mapping = new_val


def main():
    global use_head_mapping, use_dex_wrist_mapping, use_stretch_gripper_mapping
    xbox_controller = xc.XboxController()
    xbox_controller.start()
    check_usb_devices(wait_timeout=5)
    robot = rb.Robot()
    try:
        robot.startup()
        print("Using key mapping for tool: %s" % robot.end_of_arm.name)
        if robot.end_of_arm.name == "tool_none":
            use_head_mapping = True
            use_stretch_gripper_mapping = False
            use_dex_wrist_mapping = False

        if robot.end_of_arm.name == "tool_stretch_gripper":
            use_head_mapping = True
            use_stretch_gripper_mapping = True
            use_dex_wrist_mapping = False

        if robot.end_of_arm.name == "tool_stretch_dex_wrist":
            use_head_mapping = False
            use_stretch_gripper_mapping = True
            use_dex_wrist_mapping = True

        robot.pimu.trigger_beep()
        robot.push_command()
        time.sleep(0.5)

        robot.pimu.trigger_beep()
        robot.push_command()
        time.sleep(0.5)

        while True:
            controller_state = xbox_controller.get_state()
            if not robot.is_calibrated():
                manage_calibration(robot, controller_state)
            else:
                manage_base(robot, controller_state)
                manage_lift_arm(robot, controller_state)
                manage_end_of_arm(robot, controller_state)
                manage_head(robot, controller_state)
                manage_stow(robot, controller_state)
            manage_shutdown(robot, controller_state)
            robot.push_command()
            time.sleep(0.05)
    except (ThreadServiceExit, KeyboardInterrupt, SystemExit):
        robot.stop()
        xbox_controller.stop()


if __name__ == "__main__":
    main()

import pytest
import time

import numpy as np
import mrp

from home_robot.client import LocalHelloRobot


@pytest.fixture()
def home_robot_stack():
    mrp.import_msetup("../src/home_robot/agent")
    mrp.import_msetup("../src/home_robot/sim/msetup_simple_sim.py")

    mrp.cmd.up("roscore")
    mrp.cmd.up("fake_stretch")
    mrp.cmd.up("state_estimator")
    mrp.cmd.up("goto_controller")


@pytest.fixture()
def robot():
    return LocalHelloRobot()


def test_goto(home_robot_stack, robot):
    xyt_goal = [0.3, 0.2, 0.1]

    # Activate goto controller & set goal
    robot.toggle_controller()
    robot.set_goal(xyt_goal)

    # Wait for robot to reach goal
    time.sleep(2)

    # Check that robot is at goal
    xyt_new = robot.get_base_state()
    assert np.allclose(xyt_new, xyt_goal, atol=3e-3)  # 3mm

    # Down processes
    mrp.cmd.down()


if __name__ == "__main__":
    mrp.main()

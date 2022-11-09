import mrp

from home_robot.utils.mrp.envs import control_env

# State estimation node
mrp.process(
    name="state_estimator",
    runtime=mrp.Conda(
        shared_env=control_env,
        run_command=[
            "python3",
            "-m",
            "home_robot.agent.localization.state_estimator_node",
        ],
    ),
)

# Continuous controller node
mrp.process(
    name="goto_controller",
    runtime=mrp.Conda(
        shared_env=control_env,
        run_command=["python3", "-m", "home_robot.agent.control.goto_controller_node"],
    ),
)

if __name__ == "__main__":
    mrp.main()

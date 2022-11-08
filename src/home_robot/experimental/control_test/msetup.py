import mrp

from home_robot.utils.mrp.envs import control_env

mrp.process(
    name="user_interface",
    runtime=mrp.Conda(
        shared_env=control_env,
        run_command=[
            "python3",
            "-m",
            "home_robot.experimental.control_test.user_interface",
        ],
    ),
)

if __name__ == "__main__":
    mrp.main()

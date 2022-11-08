import mrp

from home_robot.utils.mrp.envs import control_env

mrp.process(
    name="stretch_cli",
    runtime=mrp.Conda(
        shared_env=control_env,
        run_command=[
            "python3",
            "-m",
            "home_robot.experimental.control_test.user_local_cli",
        ],
    ),
)

if __name__ == "__main__":
    mrp.main()

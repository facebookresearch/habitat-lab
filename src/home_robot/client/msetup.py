import mrp

from home_robot.utils.mrp.envs import control_env

mrp.process(
    name="local_cli",
    runtime=mrp.Conda(
        shared_env=control_env,
        run_command=["python3", "-m", "home_robot.client.local_hello_robot"],
    ),
)

if __name__ == "__main__":
    mrp.main()

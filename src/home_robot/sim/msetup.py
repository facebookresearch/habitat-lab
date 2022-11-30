import mrp

from home_robot.utils.mrp.envs import control_env

mrp.process(
    name="roscore_sim",
    runtime=mrp.Conda(
        shared_env=control_env,
        run_command=["python3", "-m", "home_robot.utils.mrp.roscore"],
    ),
)

mrp.process(
    name="fake_stretch",
    runtime=mrp.Conda(
        shared_env=control_env,
        run_command=["python3", "-m", "home_robot.sim.fake_stretch_robot"],
    ),
)

# Metaprocs

mrp.process(name="simple_sim", deps=["roscore", "fake_stretch"])

if __name__ == "__main__":
    mrp.main()

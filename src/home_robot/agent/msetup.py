import mrp

pip_deps = ["numpy", "scipy", "sophuspy"]

control_shared_env = mrp.Conda.SharedEnv(
    name="stretch_control_env",
    channels=["conda-forge", "robostack"],
    use_mamba=True,
    dependencies=[
        "python=3.8" "cmake",
        "pybind11",
        "ros-noetic-desktop",
        {"pip": pip_deps},
    ],
)

# State estimation node
mrp.process(
    name="state_estimator",
    runtime=mrp.Conda(
        shared_env=control_shared_env,
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
        shared_env=control_shared_env,
        run_command=["python3", "-m", "home_robot.agent.control.goto_controller_node"],
    ),
)

# Hector SLAM (TODO: contain within Conda?)
mrp.process(
    name="hector_slam",
    runtime=mrp.Host(run_command=["roslaunch", "stretch_hector_slam.launch"]),
)

if __name__ == "__main__":
    mrp.main()

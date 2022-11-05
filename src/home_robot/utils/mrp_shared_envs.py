import mrp

pip_deps = ["numpy", "scipy", "sophuspy"]

control_env = mrp.Conda.SharedEnv(
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

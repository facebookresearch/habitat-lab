import mrp

mrp.process(
    name="roscore_hw",
    runtime=mrp.Host(run_command=["python3", "-m", "home_robot.utils.mrp.roscore"]),
)

# Launches these nodes:
# - stretch_core/launch/stretch_driver.launch
# - stretch_core/launch/rplidar.launch
# - stretch_core/launch/stretch_scan_matcher.launch
mrp.process(
    name="stretch_core",
    runtime=mrp.Host(run_command=["roslaunch", "stretch_laser_odom_base.launch"]),
)

# Hector SLAM
mrp.process(
    name="stretch_hector_slam",
    runtime=mrp.Host(run_command=["roslaunch", "stretch_hector_slam.launch"]),
)

if __name__ == "__main__":
    mrp.main()

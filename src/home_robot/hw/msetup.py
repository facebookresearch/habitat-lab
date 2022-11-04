import mrp

# Launches these nodes:
# - stretch_core/launch/stretch_driver.launch
# - stretch_core/launch/rplidar.launch
# - stretch_core/launch/stretch_scan_matcher.launch
mrp.process(
    name="stretch_core",
    runtime=mrp.Host(run_command=["roslaunch", "stretch_laser_odom_base.launch"]),
)

if __name__ == "__main__":
    mrp.main()

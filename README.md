# Home Robot

## References

- [cpaxton/home_robot](https://github.com/cpaxton/home_robot)
  - Chris' repo for controlling stretch
- [facebookresearch/fairo](https://github.com/facebookresearch/fairo)
  - Robotics platform with a bunch of different stuff
  - [polymetis](https://github.com/facebookresearch/fairo/tree/main/polymetis): Contains Torchscript controllers useful for exposing low-level control logic to the user side.
  - [Meta Robotics Platform(MRP)](https://github.com/facebookresearch/fairo/tree/main/mrp): Useful for launching & managing multiple processes within their own sandboxes (to prevent dependency conflicts).
  - The [perception](https://github.com/facebookresearch/fairo/tree/main/perception) folder contains a bunch of perception related modules
    - Polygrasp: A grasping library that uses GraspNet to generate grasps and Polymetis to execute them.
    - iphone_reader: iPhone slam module.
    - realsense_driver: A thin realsense wrapper
  - [droidlet/lowlevel/hello_robot](https://github.com/facebookresearch/fairo/tree/main/droidlet/lowlevel/hello_robot)
  - Austin's branch: austinw/hello_goto_odom
- [hello-robot/stretch_body](https://github.com/hello-robot/stretch_body)
  - Base API for interacting with the Stretch robot
  - Some scripts for interacting with the Stretch
- [hello-robot/stretch_firmware](https://github.com/hello-robot/stretch_firmware)
  - Arduino firmware for the Stretch
- [hello-robot/stretch_ros](https://github.com/hello-robot/stretch_ros)
  - Builds on top of stretch_body
  - ROS-related code for Stretch
- [hello-robot/stretch_web_interface](https://github.com/hello-robot/stretch_ros2)
  - Development branch for ROS2
- [hello-robot/stretch_web_interface](https://github.com/hello-robot/stretch_web_interface)
  - Web interface for teleoping Stretch
- [RoboStack/ros-noetic](https://github.com/RoboStack/ros-noetic)
  - Conda stream with ROS binaries
- [codekansas/strech-robot](https://github.com/codekansas/stretch-robot)
  - Some misc code for interacting with RealSense camera, streaming


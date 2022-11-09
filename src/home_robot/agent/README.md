Platform-agnostic robot control code that
1. Consumes / logs sensor data
2. Computes action commands

## Instructions
```
mrp up
```
This launches:
- State estimation node
- Continuous controller node

## Notes

TODO migrate here:
- Control stack in `fairo/droidlet/lowlevel`
    - Austin's velocity control logic
    - SLAM & odom localization services
    - Planning services (map builder, FMMPlanner)
- Most of cpaxton/home_robot

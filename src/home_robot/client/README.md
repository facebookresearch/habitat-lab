## Local cli

To launch local cli:
```sh
mrp up local_cli --attach
```

Available commands:
```py
robot.get_base_state()
robot.toggle_controller()
robot.toggle_yaw_tracking()
robot.set_goal(xyt: list)
robot.set_velocity(v: float, w: float)
```


## Todos
- Teleoperation client?
- Minimal, pip-installable Pyro4 client to connect to the robot from any platform and perform simple control / visualization

from habitat_hitl.core.key_mapping import KeyCode, MouseButton
from habitat_hitl.environment.camera_helper import CameraHelper
import magnum as mn

class KeyboardInputs:
    def __init__(self, app_service, cursor_pos, camera_move_speed=0.1, robot = None):
        
        self._app_service = app_service
        self._cursor_pos = cursor_pos
        self._camera_helper = CameraHelper(
            self._app_service.hitl_config, self._app_service.gui_input
        )
        self._camera_helper.update(self._cursor_pos, 0.0)
        self.camera_move_speed = camera_move_speed
        self._robot = robot
        self._moving_robot = False
        self._sim = app_service.sim

    
    def handle_key_press(self ):
        """Handles key press events for the robot teleoperation. 
        Currently it 
        - moves the cursor position.
        - Switches between whether moving the robot through keyboard is allowed/disallowed mode
        - Moves the robot via keyboard or teleport it to a random position."""
        # update cursor positions
        self._update_cursor_position()

        # Toggle robot movement via keyboard input
        if self._app_service.gui_input.get_key(KeyCode.SPACE):
            self._moving_robot = not self._moving_robot

        # update robot positions
        if self._robot is not None and self._moving_robot:
            
            self._move_robot_on_navmesh()

            # sample new placement for robot on navmesh if that is requested.
            if self._app_service.gui_input.get_key_down(KeyCode.M):
                self._robot.place_robot(
                    self._sim.pathfinder.get_random_navigable_point()
                )


    def _update_cursor_position(self):
        """
        Updates the position of the camera focus point when keys are pressed.
        Equivalent to "walking" in the scene.
        """
        gui_input = self._app_service.gui_input
        y_speed = 0.1
        # NOTE: cursor elevation is always controllable
        if gui_input.get_key(KeyCode.Z):
            self._cursor_pos.y -= y_speed
        if gui_input.get_key(KeyCode.X):
            self._cursor_pos.y += y_speed
        
        # manual cursor control
        xz_forward = self._camera_helper.get_xz_forward()
        xz_right = mn.Vector3(-xz_forward.z, 0.0, xz_forward.x)
        speed = (
            self.camera_move_speed
            * self._camera_helper.cam_zoom_dist
        )
        if gui_input.get_key(KeyCode.W):
            self._cursor_pos += xz_forward * speed
        if gui_input.get_key(KeyCode.S):
            self._cursor_pos -= xz_forward * speed
        if gui_input.get_key(KeyCode.D):
            self._cursor_pos += xz_right * speed
        if gui_input.get_key(KeyCode.A):
            self._cursor_pos -= xz_right * speed

    
    def _move_robot_on_navmesh(self):
        """
        Handles key press updates the robot on the navmesh.
        """

        gui_input = self._app_service.gui_input
        speed = self.camera_move_speed
        if self._robot is not None:

            start = self._robot.ao.translation
            end = mn.Vector3(start)
            r_speed = 0.05

            # Handle Robot Tramslation
            if gui_input.get_key(KeyCode.I):
                end = end + self._robot.ao.transformation.transform_vector(
                    mn.Vector3(speed, 0, 0)
                )
            if gui_input.get_key(KeyCode.K):
                end = end + self._robot.ao.transformation.transform_vector(
                    mn.Vector3(-speed, 0, 0)
                )

            if start != end:
                self._robot.ao.translation = self._sim.pathfinder.try_step(
                    start, end
                )
            
            # Handle Robot Rotation
            if gui_input.get_key(KeyCode.L):
                r = mn.Quaternion.rotation(
                    mn.Rad(-r_speed), mn.Vector3(0, 1, 0)
                )
                self._robot.ao.rotation = r * self._robot.ao.rotation
            if gui_input.get_key(KeyCode.J):
                r = mn.Quaternion.rotation(
                    mn.Rad(r_speed), mn.Vector3(0, 1, 0)
                )
                self._robot.ao.rotation = r * self._robot.ao.rotation

            
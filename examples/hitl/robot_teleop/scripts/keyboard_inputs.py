from habitat_hitl.core.key_mapping import KeyCode, MouseButton
from habitat_hitl.environment.camera_helper import CameraHelper
import magnum as mn
from typing import Any, Dict
import habitat.sims.habitat_simulator.sim_utilities as sutils
from scripts.utilities import (import_robot, build_navmesh_lines, 
                               add_object_at, get_mouse_cast, remove_object)

class KeyboardInputs:
    def __init__(self, app_service, cursor_pos, _app_cfg, robot = None) -> None:
        
        self._app_service = app_service
        self._cursor_pos = cursor_pos
        self._app_cfg = _app_cfg


        self._camera_helper = CameraHelper(
            self._app_service.hitl_config, self._app_service.gui_input
        )
        self._camera_helper.update(self._cursor_pos, 0.0)
        self.camera_move_speed = self._app_cfg.camera_move_speed
        self.robot = robot
        self._moving_robot = False
        self._sim = app_service.sim
        self.navmesh_lines = None

        self.added_object_ids = []
        self.mouse_cast_results = None

        self._exit_app = False
        self._hide_gui = False

    @property
    def exit_app(self) -> bool:
        return self._exit_app
    
    @property
    def hide_gui(self) -> bool:
        return self._hide_gui
    
    @property
    def cursor_pos(self) -> mn.Vector3:
        return self._cursor_pos
    
    
    def handle_key_press(self, camera_helper = None) -> None:
        """Handles key press events for the robot teleoperation. 
        Currently it 
        - moves the cursor position.
        - Switches between whether moving the robot through keyboard is allowed/disallowed mode
        - Moves the robot via keyboard or teleport it to a random position."""


        # update cursor positions
        self._update_cursor_position(camera_helper)

        
        # handle application characteristics (exit, hide gui, etc.)
        self._handle_application_characteristics()

        # handle navmesh features
        self._handle_navmesh_features()

        # handle robot characteristics (teleoperation, etc.)
        self._handle_robot_characteristics()

        # Handle object placement and removal
        self._handle_object_placement()

    def _handle_object_placement(self) -> None:
        """
        Handles object placement and removal.
        """
        gui_input = self._app_service.gui_input
        self.mouse_cast_results = get_mouse_cast(self)


        # Place objects
        if gui_input.get_key_down(KeyCode.Y):


            if self._app_cfg.ycb_objects.use_cursor:
                # insert an object at the mouse raycast position. Ask for the index position of object.
                idx = input(
                    "Enter the index of the object to add 0 - " + str(len(self._app_cfg.ycb_objects.names) - 1 ) + " > ")
                
                # check if the index is a number
                if not idx.isnumeric():
                    print("Index must be a number.")
                    return
                # check if the index is a positive number
                if idx == "":
                    print("Index must be a number.")
                    return
                
                # check if index is within the range of the object list
                if int(idx) < 0 or int(idx) >= len(self._app_cfg.ycb_objects.names): 
                    print("Invalid index for object to add.")
                    return

                obj_shortname = self._app_cfg.ycb_objects.names[int(idx)]
                obj_template_handle = list(
                    self._sim.get_object_template_manager()
                    .get_templates_by_handle_substring(obj_shortname)
                    .keys()
                )[0]
                new_obj = add_object_at(self, obj_template_handle)
                self.added_object_ids.append(new_obj.object_id)
                obj_size_down = sutils.get_obj_size_along(
                    self._sim, new_obj.object_id, mn.Vector3(0, -1, 0)
                )
                placement_position = self.mouse_cast_results.hits[
                    0
                ].point + mn.Vector3(0, obj_size_down[0], 0)
                new_obj.translation = placement_position

                # TO DO: ensure habitat gets these from the object itself instead of manual setting.
                new_obj.mass = 0.01
                new_obj.rolling_friction_coefficient = 5
                new_obj.spinning_friction_coefficient = 5
                new_obj.friction_coefficient = 5

            else:

                if len(self._app_cfg.ycb_objects.names) != len(
                    self._app_cfg.ycb_objects.positions
                ):
                    print("YCB object names and positions must be the same length.")
                    return

                
                for idx in range(len(self._app_cfg.ycb_objects.names)):
                    obj_shortname = self._app_cfg.ycb_objects.names[idx]
                    obj_template_handle = list(
                        self._sim.get_object_template_manager()
                        .get_templates_by_handle_substring(obj_shortname)
                        .keys()
                    )[0]

                    new_obj = add_object_at(self, obj_template_handle)
                    self.added_object_ids.append(new_obj.object_id)
                    obj_size_down = sutils.get_obj_size_along(
                    self._sim, new_obj.object_id, mn.Vector3(0, -1, 0)
                    )

                    position = self._app_cfg.ycb_objects.positions[idx]
                    new_obj.translation = mn.Vector3(position[0], position[1], position[2])
                    # TO DO: ensure habitat gets these from the object itself instead of manual setting.
                    new_obj.mass = 0.01
                    new_obj.rolling_friction_coefficient = 5
                    new_obj.spinning_friction_coefficient = 5
                    new_obj.friction_coefficient = 5

        # Remove objects
        if gui_input.get_key_down(KeyCode.U):
            remove_object(self, self.mouse_cast_results.hits[0].object_id)
            self.added_object_ids.remove(self.mouse_cast_results.hits[0].object_id)




    def _handle_application_characteristics(self ) -> None:
        """
        Handles application characteristics such as:
        - Exiting the application
        - Hiding the GUI
        """

        gui_input = self._app_service.gui_input
        if gui_input.get_key_down(KeyCode.ESC):
            self._exit_app = True

        if gui_input.get_key_down(KeyCode.H):
            self._hide_gui = not self._hide_gui



    def _handle_navmesh_features(self) -> None:
        """
        Handles navmesh features such as:
        - Recomputing the navmesh
        - Building navmesh lines for visualization
        """
        gui_input = self._app_service.gui_input
        # navmesh features
        if gui_input.get_key_down(KeyCode.N):
            if self.navmesh_lines is not None:
                self.navmesh_lines = None
            else:
                self.navmesh_lines = build_navmesh_lines(self)
    
    
    def _handle_robot_characteristics(self) -> None:

        """
        Handles robot characteristics such as:
        - Teleporting the robot to a random position on the navmesh
        - Moving the robot via keyboard input
        - Hot reloading the robot
        - Saving/loading robot poses to/from cache
        """

        gui_input = self._app_service.gui_input

        # Toggle robot movement via keyboard input
        if self._app_service.gui_input.get_key(KeyCode.SPACE):
            self._moving_robot = not self._moving_robot

        # update robot positions
        if self.robot is not None and self._moving_robot:
            
            self._move_robot_on_navmesh()

         # sample new placement for robot on navmesh if that is requested.
        if self._app_service.gui_input.get_key_down(KeyCode.M) and self.robot is not None:
            self.robot.place_robot(
                self._sim.pathfinder.get_random_navigable_point()
            )


        # remove the current robot and reload from URDF
        # I.E., Hot-reload robot
        if gui_input.get_key_down(KeyCode.T):
            self.robot.clean()
            self.robot = import_robot(self)


        
        # robot pose caching and loading
        # P saves the current robot pose to cache
        # O loads the robot pose from cache 
        if (gui_input.get_key_down(KeyCode.P) or gui_input.get_key_down(KeyCode.O)) and self.robot is not None:
            # cache the current pose
            
            configuration_subset_name = input(
                " Enter the configuration subset key. (Press ENTER without input for full pose.) >"
            )
            if (
                configuration_subset_name != ""
                and configuration_subset_name not in self.robot.pos_subsets
            ):
                print(
                    f" Invalid configuration subset name. Defined subsets are: {self.robot.pos_subsets.keys()}"
                )
                return
            cached_pose_name = input(
                "Enter the desired cache key name of the pose > "
            )
            
            if cached_pose_name == "":
                print("No cache key name entered, defaulted to 'default'.")
                cached_pose_name = "default"


            if configuration_subset_name == "":
                
                if gui_input.get_key_down(KeyCode.P):
                    # Save the pose
                    print("Saving pose to cache:")
                    self.robot.cache_pose(pose_name=cached_pose_name)
                    print(
                        f"Pose cached with name {cached_pose_name} for full robot."
                    )

                elif gui_input.get_key_down(KeyCode.O):
                    # Load the pose
                    print("Loading pose from cache:")
                    self.robot.set_cached_pose(
                    pose_name=cached_pose_name,
                    set_motor_targets=self.robot.using_joint_motors,
                    set_positions=not self.robot.using_joint_motors,
                )
                    print(
                        f"Pose loaded with name {cached_pose_name} for full robot."
                    )
            else:

                if gui_input.get_key_down(KeyCode.P):
                    # Save the pose
                    print("Saving pose to cache:")
                    self.robot.pos_subsets[configuration_subset_name].cache_pose(
                        pose_name=cached_pose_name
                    )
                    print(
                        f"Pose cached with name {cached_pose_name} for subset {configuration_subset_name}"
                    )

                elif gui_input.get_key_down(KeyCode.O):
                    # Load the pose
                    print("Loading pose from cache:")
                    self.robot.pos_subsets[
                    configuration_subset_name].set_cached_pose(
                    pose_name=cached_pose_name,
                    set_motor_targets=self.robot.using_joint_motors,
                    set_positions=not self.robot.using_joint_motors,
                     )
                    print(
                        f"Pose loaded with name {cached_pose_name} for subset {configuration_subset_name}"
                    )

        
                
        

    def _update_cursor_position(self, camera_helper = None) -> None:
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
        
        
        if camera_helper is not None:
            self._camera_helper = camera_helper
        else:
            print("No Camera Object Passed. Movement will be relative to the initialized camera pose.")
        
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

    
    def _move_robot_on_navmesh(self) -> None:
        """
        Handles key press updates the robot on the navmesh.
        """

        gui_input = self._app_service.gui_input
        speed = self.camera_move_speed
        if self.robot is not None:

            start = self.robot.ao.translation
            end = mn.Vector3(start)
            r_speed = 0.05

            # Handle Robot Tramslation
            if gui_input.get_key(KeyCode.I):
                end = end + self.robot.ao.transformation.transform_vector(
                    mn.Vector3(speed, 0, 0)
                )
            if gui_input.get_key(KeyCode.K):
                end = end + self.robot.ao.transformation.transform_vector(
                    mn.Vector3(-speed, 0, 0)
                )

            if start != end:
                self.robot.ao.translation = self._sim.pathfinder.try_step(
                    start, end
                )
            
            # Handle Robot Rotation
            if gui_input.get_key(KeyCode.L):
                r = mn.Quaternion.rotation(
                    mn.Rad(-r_speed), mn.Vector3(0, 1, 0)
                )
                self.robot.ao.rotation = r * self.robot.ao.rotation
            if gui_input.get_key(KeyCode.J):
                r = mn.Quaternion.rotation(
                    mn.Rad(r_speed), mn.Vector3(0, 1, 0)
                )
                self.robot.ao.rotation = r * self.robot.ao.rotation

            
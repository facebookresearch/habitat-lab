#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from isaacsim import SimulationApp
import multiprocessing

def do_isaaclab_imports():
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCuboid
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims.rigid_prim import RigidPrim
    from omni.isaac.core.prims.rigid_prim_view import RigidPrimView

    from pxr import Usd, UsdPhysics, PhysxSchema
    import omni.physx.scripts.utils as physxUtils
    globals().update(locals())

import numpy as np
import time
from scipy.spatial.transform import Rotation as R

class IsaacSimTest:
    def __init__(self, headless=False, worker_name="0"):
        
        self._worker_name = worker_name

        experience_path = "./my_isaacsim_experience.kit"
        simulation_app = SimulationApp({"headless": headless}, experience_path) # we can also run as headless.

        do_isaaclab_imports()

        world = World()
        self._world = world
        world.scene.add_default_ground_plane()
        fancy_cube =  world.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube",
                name="fancy_cube",
                position=np.array([0, 0, 1.0]),
                scale=np.array([0.5015, 0.5015, 0.5015]),
                color=np.array([0, 0, 1.0]),
            ))

        asset_path = "./data/usd/converted_robot.usda"
        robot_prim_path = "/World/Spot"

        add_reference_to_stage(usd_path=asset_path, prim_path=robot_prim_path)

        robot_prim = world.stage.GetPrimAtPath(robot_prim_path)

        if not robot_prim.IsValid():
            raise ValueError(f"Prim at {robot_prim_path} is not valid.")

        # Traverse only the robot's prim hierarchy
        for prim in Usd.PrimRange(robot_prim):

            if prim.HasAPI(UsdPhysics.DriveAPI):
                # Access the existing DriveAPI
                drive_api = UsdPhysics.DriveAPI(prim, "angular")
                if drive_api:

                    # Modify drive parameters
                    drive_api.GetStiffnessAttr().Set(10.0)  # Position gain
                    drive_api.GetDampingAttr().Set(0.1)     # Velocity gain
                    drive_api.GetMaxForceAttr().Set(1000)  # Maximum force/torque

                drive_api = UsdPhysics.DriveAPI(prim, "linear")
                if drive_api:

                    drive_api = UsdPhysics.DriveAPI.Get(prim, "linear")
                    drive_api.GetStiffnessAttr().Set(1000)  # Example for linear stiffness

            if prim.HasAPI(UsdPhysics.RigidBodyAPI):

                # UsdPhysics.RigidBodyAPI doesn't support damping but PhysxRigidBodyAPI does
                if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)
                else:
                    physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

                # todo: decide hard-coded values here
                physx_api.CreateLinearDampingAttr(50.0)
                physx_api.CreateAngularDampingAttr(10.0)

        # todo: investigate if this is needed for kinematic base
        self.scale_prim_mass_and_inertia(f"{robot_prim_path}/base", 100.0)

        self._robot = world.scene.add(Robot(prim_path=robot_prim_path, name="my_robot"))
        self._robot_controller = self._robot.get_articulation_controller()

        # Resetting the world needs to be called before querying anything related to an articulation specifically.
        # Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
        world.reset()
        # world.step(render=False)
        if not headless:
            world.pause()

        self._step_count = 0

        # Add the callback to the world
        world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)

        world.add_physics_callback("fix_base", callback_fn=self.fix_base)

        while simulation_app.is_running():
            # Update the simulation app
            simulation_app.update()
            # Add a short sleep to avoid maxing out CPU usage
            time.sleep(0.01)


        simulation_app.close() # close Isaac Sim


    # Define a physics callback to apply actions
    def send_robot_actions(self, step_size):
        self._step_count += 1

        if self._step_count % 200 == 0:
            # Apply random joint velocities to the robot's first two joints
            # velocities = 5 * np.ones((20,))
            # robot_controller.apply_action(
            #     ArticulationAction(joint_positions=None, joint_efforts=None, joint_velocities=velocities)
            # )
            positions = np.random.uniform(-0.5, 0.5, 20)
            self._robot_controller.apply_action(
                ArticulationAction(joint_positions=positions, joint_efforts=None, joint_velocities=None)
            )
            # random_efforts = 5 * np.ones((20,))
            # robot_controller.apply_action(
            #     ArticulationAction(joint_positions=None, joint_efforts=random_efforts, joint_velocities=None)
            # )

            # also periodically print robot root pose
            base_pos, base_orientation = self._robot.get_world_pose()
            print(f"worker {self._worker_name} base pose: ({base_pos[0]:.4f}, {base_pos[1]:.4f}, {base_pos[2]:.4f}), ({base_orientation[0]:.4f}, {base_orientation[1]:.4f}, {base_orientation[2]:.4f}, {base_orientation[3]:.4f})")

        
    def scale_prim_mass_and_inertia(self, path, scale):

        prim = self._world.stage.GetPrimAtPath(path)
        assert prim.HasAPI(UsdPhysics.MassAPI)
        mass_api = UsdPhysics.MassAPI(prim)
        mass_api.GetMassAttr().Set(mass_api.GetMassAttr().Get() * scale)
        mass_api.GetDiagonalInertiaAttr().Set(mass_api.GetDiagonalInertiaAttr().Get() * scale)


    def fix_base_orientation_via_angular_vel(self, step_size):

        _, base_orientation = self._robot.get_world_pose()

        # Constants
        max_angular_velocity = 3.0  # Maximum angular velocity (rad/s)

        # wxyz to xyzw
        base_orientation_xyzw = np.array([base_orientation[1], base_orientation[2], base_orientation[3], base_orientation[0]])
        base_rotation = R.from_quat(base_orientation_xyzw)

        # Define the local "up" axis and transform it to world space
        local_up = np.array([0, 0, 1])  # Object's local up axis (hack: -1?)
        world_up = base_rotation.apply(local_up)  # Local up in world space

        # Define the global up axis
        global_up = np.array([0, 0, 1])  # Global up direction

        # Compute the axis of rotation to align world_up to global_up
        rotation_axis = np.cross(world_up, global_up)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        # Handle special cases where no rotation is needed
        if rotation_axis_norm < 1e-6:  # Already aligned
            desired_angular_velocity = np.array([0, 0, 0])  # No correction needed
        else:
            # Normalize the rotation axis
            rotation_axis /= rotation_axis_norm

            # Compute the angle of rotation using the dot product
            rotation_angle = np.arccos(np.clip(np.dot(world_up, global_up), -1.0, 1.0))

            # Calculate the angular velocity to correct the tilt in one step
            hack_scale = 1.0
            tilt_correction_velocity = ((rotation_axis * rotation_angle) / step_size) * hack_scale

            # Cap the angular velocity to the maximum allowed value
            angular_velocity_magnitude = np.linalg.norm(tilt_correction_velocity)
            if angular_velocity_magnitude > max_angular_velocity:
                tilt_correction_velocity *= max_angular_velocity / angular_velocity_magnitude

            desired_angular_velocity = tilt_correction_velocity

        self._robot.set_angular_velocity(desired_angular_velocity)


    def fix_base_height_via_linear_vel_z(self, step_size):
        z_target = 1.0  # todo: get from navmesh or assume ground_z==0
        max_linear_vel = 3.0

        # Get the current position and velocity of the base
        base_position, _ = self._robot.get_world_pose()

        # Extract the vertical position and velocity
        z_current = base_position[2]

        # Compute the position error
        position_error = z_target - z_current    

        desired_linear_vel_z = position_error / step_size
        desired_linear_vel_z = max(-max_linear_vel, min(max_linear_vel, desired_linear_vel_z))

        self._robot.set_linear_velocity([0.0, 0.0, desired_linear_vel_z])

    def fix_base(self, step_size):
        self.fix_base_height_via_linear_vel_z(step_size)
        self.fix_base_orientation_via_angular_vel(step_size)


def multiprocess_test():

    def worker_process(identifier):
        test = IsaacSimTest(headless=True, worker_name=f"{identifier}")

    processes = []
    for i in range(1):
        print(f"starting worker {i}...")
        p = multiprocessing.Process(target=worker_process, args=(i,))
        processes.append(p)
        p.start()
        print("sleeping 15s...")
        time.sleep(15)


    # Let the processes run for 30 seconds
    time.sleep(30)

    # Terminate all processes from the main process
    print("Terminating all worker processes...")
    for p in processes:
        p.terminate()
        p.join()  # Ensure the process is cleaned up

    print("All worker processes terminated.")    

if __name__ == "__main__":
    # multiprocess_test()
    test = IsaacSimTest(headless=True, worker_name="0")
import habitat
import cv2

import os
import time
import git

import magnum as mn
import matplotlib.pyplot as plt
import numpy as np
import math

import habitat_sim
from habitat_sim.utils import viz_utils as vut

# import quadruped_wrapper
import ant_robot

repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "../habitat-sim/data")

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def place_agent(sim):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    agent_state.position = [-0.15, -0.7, 1.0]
    agent_state.rotation = np.quaternion(-0.83147, 0, 0.55557, 0)
    agent = sim.initialize_agent(0, agent_state)
    return agent.scene_node.transformation_matrix()

def make_configuration():
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "NONE"

    backend_cfg.enable_physics = True

    # sensor configurations
    # Note: all sensors must have the same resolution
    # setup 2 rgb sensors for 1st and 3rd person views
    camera_resolution = [544, 720]
    sensor_specs = []

    rgba_camera_1stperson_spec = habitat_sim.CameraSensorSpec()
    rgba_camera_1stperson_spec.uuid = "rgba_camera_1stperson"
    rgba_camera_1stperson_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgba_camera_1stperson_spec.resolution = camera_resolution
    rgba_camera_1stperson_spec.position = [0.0, 0.6, 0.0]
    rgba_camera_1stperson_spec.orientation = [0.0, 0.0, 0.0]
    rgba_camera_1stperson_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(rgba_camera_1stperson_spec)

    depth_camera_1stperson_spec = habitat_sim.CameraSensorSpec()
    depth_camera_1stperson_spec.uuid = "depth_camera_1stperson"
    depth_camera_1stperson_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_camera_1stperson_spec.resolution = camera_resolution
    depth_camera_1stperson_spec.position = [0.0, 0.6, 0.0]
    depth_camera_1stperson_spec.orientation = [0.0, 0.0, 0.0]
    depth_camera_1stperson_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_camera_1stperson_spec)

    rgba_camera_3rdperson_spec = habitat_sim.CameraSensorSpec()
    rgba_camera_3rdperson_spec.uuid = "rgba_camera_3rdperson"
    rgba_camera_3rdperson_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgba_camera_3rdperson_spec.resolution = camera_resolution
    rgba_camera_3rdperson_spec.position = [0.0, 1.0, 0.3]
    rgba_camera_3rdperson_spec.orientation = [-45, 0.0, 0.0]
    rgba_camera_3rdperson_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(rgba_camera_3rdperson_spec)

    # agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])



def example():
    # Note: Use with for the example testing, doesn't need to be like this on the README

    cfg = make_configuration()
    sim = habitat_sim.Simulator(cfg)
    agent_transform = place_agent(sim)

    # get the primitive assets attributes manager
    prim_templates_mgr = sim.get_asset_template_manager()

    # get the physics object attributes manager
    obj_templates_mgr = sim.get_object_template_manager()

    # get the rigid object manager
    rigid_obj_mgr = sim.get_rigid_object_manager()

    observations = []
    count_steps = 0

    # add floor
    # build box
    cube_handle = obj_templates_mgr.get_template_handles("cube")[0]
    floor = obj_templates_mgr.get_template_by_handle(cube_handle)
    floor.scale = np.array([2.0, 0.05, 2.0])

    obj_templates_mgr.register_template(floor, "floor")
    floor_obj = rigid_obj_mgr.add_object_by_template_handle("floor")
    floor_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC

    floor_obj.translation = np.array([2.50, -1, 0.5])
    floor_obj.motion_type = habitat_sim.physics.MotionType.STATIC

    # Add ant robot
    robot_path = "data/robots/ant.urdf"

    ant = ant_robot.AntV2Robot(robot_path, sim)
    ant.reconfigure()
    ant.base_pos = mn.Vector3(2.50, 1.0, 0.2)
    ant.base_rot = math.pi / 2
    print(ant.ankle_joint_pos)

    while True:
        keystroke = cv2.waitKey(0)
        if keystroke == 27:
            break
        
        sim.step_physics(1.0 / 60.0)
        observations.append(sim.get_sensor_observations())

        if count_steps == 120:
            ant.leg_joint_pos = [0, 0, 0, 0, -0.3, 0.3, 0.3, -0.3]

        if count_steps == 180:
            ant.leg_joint_pos = [1, 1, 1, 1, -1, 1, 1, -1]
        
        if count_steps == 210:
            ant.leg_joint_pos = [0, 0, 0, 0, -1, 1, 1, -1]

        print(ant.observational_space)
        print(count_steps)
        print(keystroke)
        print("_____")

        #observations = env.step(env.action_space.sample())  # noqa: F841
        cv2.imshow("RGB", transform_rgb_bgr(observations[-1]["rgba_camera_1stperson"]))
        count_steps += 1
    print("Episode finished after {} steps.".format(count_steps))

    vut.make_video(
        observations,
        "rgba_camera_1stperson",
        "color",
        "test_ant_wrapper",
        open_vid=True,
    )
    

if __name__ == "__main__":
    example()

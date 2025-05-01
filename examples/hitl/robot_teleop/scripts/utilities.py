from hydra import compose
from scripts.robot import Robot
import habitat_sim

def recompute_navmesh(obj) -> None:
    """
    Recomputes the scene navmesh using the currently loaded robot's radius.
    """
    mt_cache = {}
    # set all Aos to STATIC so they are baked into the navmesh
    for _handle, ao in (
        obj._sim.get_articulated_object_manager()
        .get_objects_by_handle_substring()
        .items()
    ):
        if obj.robot.ao.handle != ao.handle:
            mt_cache[ao.handle] = ao.motion_type
            ao.motion_type = habitat_sim.physics.MotionType.STATIC

    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    # get height and radius from the robot
    navmesh_settings.agent_height = (
        obj.robot.robot_cfg.navmesh_height
        if obj.robot is not None
        else 1.3
    )
    navmesh_settings.agent_radius = (
        obj.robot.robot_cfg.navmesh_radius
        if obj.robot is not None
        else 0.25
    )
    navmesh_settings.include_static_objects = True
    obj._sim.recompute_navmesh(
        obj._sim.pathfinder,
        navmesh_settings,
    )
    # set all Aos to back to their previous motion types
    for _handle, ao in (
        obj._sim.get_articulated_object_manager()
        .get_objects_by_handle_substring()
        .items()
    ):
        if obj.robot.ao.handle != ao.handle:
            ao.motion_type = mt_cache[ao.handle]
    

def build_navmesh_lines(obj) -> None:
    """
    Constructs a set of debug line endpoint pairs for all edges on the navmesh.
    """
    obj.navmesh_lines = []
    # NOTE: This could be color coded by island or restricted to a particular island
    i = obj._sim.pathfinder.build_navmesh_vertex_indices()
    v = obj._sim.pathfinder.build_navmesh_vertices()
    for tri in range(int(len(i) / 3)):
        v0 = v[i[tri * 3]]
        v1 = v[i[tri * 3 + 1]]
        v2 = v[i[tri * 3 + 2]]
        obj.navmesh_lines.extend([(v0, v1), (v1, v2), (v2, v0)])
    
    return obj.navmesh_lines


def import_robot(obj) -> None:
    """
    Imports the robot defined in the yaml config.
    """
    # hot reload the config file
    robot_cfg = compose(config_name="robot_settings")
    if robot_cfg is not None:
        # initialize the robot from config
        obj.robot = Robot(obj._sim, robot_cfg)
        recompute_navmesh(obj)
        obj.robot.place_robot(
            obj._sim.pathfinder.get_random_navigable_point()
        )
        # load the grasping poses. If this line fails (i.e. integrating a new robot, comment it and record the configuration subset and poses with the same key names below)
        obj.hand_grasp_poses = [
            obj.robot.pos_subsets["left_hand"].fetch_cached_pose(
                pose_name=hand_subset_key
            )
            for hand_subset_key in ["hand_open", "grasp"]
        ]

        return obj.robot
    else:
        print("No robot configured.")
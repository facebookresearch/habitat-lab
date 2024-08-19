from typing import Optional, cast

import magnum as mn
import numpy as np

import habitat_sim
from habitat.tasks.rearrange.marker_info import MarkerInfo
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    PddlEntity,
    PddlSimInfo,
    SimulatorObjectType,
)
from habitat.tasks.rearrange.utils import (
    place_agent_at_dist_from_pos,
    rearrange_logger,
)

# TODO: Deprecate these and instead represent them as articulated object entity type.
CAB_TYPE = "cab_type"
FRIDGE_TYPE = "fridge_type"


def is_robot_hold_match(
    robot: PddlEntity,
    sim_info: PddlSimInfo,
    hold_state: bool,
    obj: Optional[PddlEntity] = None,
) -> bool:
    """
    Check if the robot is holding the desired object in the desired hold state.
    :param hold_state: True if the robot should be holding the object.
    """

    robot_id = cast(
        int,
        sim_info.search_for_entity(robot),
    )
    grasp_mgr = sim_info.sim.get_agent_data(robot_id).grasp_mgr

    if hold_state:
        if obj is not None:
            # Robot must hold specific object.
            obj_idx = cast(int, sim_info.search_for_entity(obj))
            abs_obj_id = sim_info.sim.scene_obj_ids[obj_idx]
            return grasp_mgr.snap_idx == abs_obj_id
        else:
            # Robot can hold any object.
            return grasp_mgr.snap_idx != None
    else:
        # Robot must hold no object.
        return grasp_mgr.snap_idx == None


def set_robot_holding(
    robot: PddlEntity,
    sim_info: PddlSimInfo,
    hold_state: bool,
    obj: Optional[PddlEntity] = None,
) -> None:
    robot_id = cast(
        int,
        sim_info.search_for_entity(robot),
    )
    sim = sim_info.sim
    agent_data = sim.get_agent_data(robot_id)
    # Set the snapped object information
    if not hold_state and agent_data.grasp_mgr.is_grasped:
        agent_data.grasp_mgr.desnap(True)
    elif hold_state:
        if obj is None:
            raise ValueError(
                f"If setting hold state {hold_state=}, must set object"
            )
        # Swap objects to the desired object.
        obj_idx = cast(int, sim_info.search_for_entity(obj))
        agent_data.grasp_mgr.desnap(True)
        sim.internal_step(-1)
        agent_data.grasp_mgr.snap_to_obj(sim.scene_obj_ids[obj_idx])
        sim.internal_step(-1)


def is_inside(
    obj: PddlEntity, recep: PddlEntity, sim_info: PddlSimInfo
) -> bool:
    """
    Check if an entity is inside the receptacle.
    """

    assert sim_info.check_type_matches(
        recep, SimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value
    ), f"Bad type {recep=}"

    entity_pos = sim_info.get_entity_pos(obj)
    check_marker = cast(
        MarkerInfo,
        sim_info.search_for_entity(recep),
    )
    # Hack to see if an object is inside the fridge.
    if sim_info.check_type_matches(recep, FRIDGE_TYPE):
        global_bb = habitat_sim.geo.get_transformed_bb(
            check_marker.ao_parent.aabb,
            check_marker.ao_parent.transformation,
        )
    else:
        bb = check_marker.link_node.cumulative_bb
        global_bb = habitat_sim.geo.get_transformed_bb(
            bb, check_marker.link_node.transformation
        )

    return global_bb.contains(entity_pos)


def is_robot_at_position(
    at_entity,
    sim_info,
    dist_thresh: float,
    robot=None,
    angle_thresh: Optional[float] = None,
):
    if robot is None:
        robot = sim_info.sim.get_agent_data(None).articulated_agent
    else:
        robot_id = cast(
            int,
            sim_info.search_for_entity(robot),
        )
        robot = sim_info.sim.get_agent_data(robot_id).articulated_agent
    targ_pos = sim_info.get_entity_pos(at_entity)

    # Get the base transformation
    T = robot.base_transformation
    # Do transformation
    pos = T.inverted().transform_point(targ_pos)
    # Project to 2D plane (x,y,z=0)
    pos[2] = 0.0

    # Compute distance
    dist = np.linalg.norm(pos)

    # Unit vector of the pos
    pos = pos.normalized()
    # Define the coordinate of the robot
    pos_robot = np.array([1.0, 0.0, 0.0])
    # Get the angle
    angle = np.arccos(np.dot(pos, pos_robot))

    # Check the distance threshold.
    if dist > dist_thresh:
        return False

    # Check for the angle threshold
    if angle_thresh is not None and np.abs(angle) > angle_thresh:
        return False

    return True


def set_robot_position(
    at_entity: PddlEntity,
    sim_info: PddlSimInfo,
    dist_thresh: float,
    robot: Optional[PddlEntity] = None,
    filter_colliding_states: bool = True,
    angle_noise: float = 0.0,
    num_spawn_attempts: int = 200,
):
    """
    Set the robot transformation to be within `dist_thresh` of `at_entity`.
    """

    sim = sim_info.sim
    if robot is None:
        agent_data = sim.get_agent_data(None)
    else:
        robot_id = cast(
            int,
            sim_info.search_for_entity(robot),
        )
        agent_data = sim.get_agent_data(robot_id)
    targ_pos = sim_info.get_entity_pos(at_entity)

    # Place some distance away from the object.
    start_pos, start_rot, was_fail = place_agent_at_dist_from_pos(
        target_position=targ_pos,
        rotation_perturbation_noise=angle_noise,
        distance_threshold=dist_thresh,
        sim=sim,
        num_spawn_attempts=num_spawn_attempts,
        filter_colliding_states=filter_colliding_states,
        agent=agent_data.articulated_agent,
    )
    agent_data.articulated_agent.base_pos = start_pos
    agent_data.articulated_agent.base_rot = start_rot
    if was_fail:
        rearrange_logger.error("Failed to place the robot.")

    # We teleported the agent. We also need to teleport the object the agent was holding.
    agent_data.grasp_mgr.update_object_to_grasp()


def is_object_at(
    obj: PddlEntity,
    at_entity: PddlEntity,
    sim_info: PddlSimInfo,
    dist_thresh: float,
) -> bool:
    """
    Checks if an object entity is logically at another entity. At an object
    means within a threshold of that object. At a receptacle means on the
    receptacle. At a articulated receptacle means inside of it.
    """

    entity_pos = sim_info.get_entity_pos(obj)

    if sim_info.check_type_matches(
        at_entity, SimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value
    ):
        # Object is rigid and target is receptacle, we are checking if
        # an object is inside of a receptacle.
        return is_inside(obj, at_entity, sim_info)
    elif sim_info.check_type_matches(
        at_entity, SimulatorObjectType.GOAL_ENTITY.value
    ) or sim_info.check_type_matches(
        at_entity, SimulatorObjectType.MOVABLE_ENTITY.value
    ):
        # Is the target `at_entity` a movable or goal entity?
        targ_idx = cast(
            int,
            sim_info.search_for_entity(at_entity),
        )
        idxs, pos_targs = sim_info.sim.get_targets()
        targ_pos = pos_targs[list(idxs).index(targ_idx)]

        dist = float(np.linalg.norm(entity_pos - targ_pos))
        return dist < dist_thresh
    elif sim_info.check_type_matches(
        at_entity, SimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value
    ):
        # TODO: Fix this logic to be using
        # habitat/sims/habitat_simulator/sim_utilities.py
        recep = cast(mn.Range3D, sim_info.search_for_entity(at_entity))
        return recep.contains(entity_pos)
    else:
        raise ValueError(
            f"Got unexpected combination of {obj} and {at_entity}"
        )


def set_object_at(
    obj: PddlEntity,
    at_entity: PddlEntity,
    sim_info: PddlSimInfo,
    recep_place_shrink_factor: float = 0.8,
) -> None:
    """
    Sets a movable PDDL entity to match the transformation of a desired
    `at_entity` which can be a receptacle or goal.

    :param recep_place_shrink_factor: How much to shrink the size of the
        receptacle by when placing the entity on a receptacle.
    """

    sim = sim_info.sim

    # The source object must be movable.
    if not sim_info.check_type_matches(
        obj, SimulatorObjectType.MOVABLE_ENTITY.value
    ):
        raise ValueError(f"Got unexpected obj {obj}")

    if sim_info.check_type_matches(
        at_entity, SimulatorObjectType.GOAL_ENTITY.value
    ):
        targ_idx = cast(
            int,
            sim_info.search_for_entity(at_entity),
        )
        all_targ_idxs, pos_targs = sim.get_targets()
        targ_pos = pos_targs[list(all_targ_idxs).index(targ_idx)]
        set_T = mn.Matrix4.translation(targ_pos)
    elif sim_info.check_type_matches(
        at_entity, SimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value
    ):
        # Place object on top of receptacle.
        recep = cast(mn.Range3D, sim_info.search_for_entity(at_entity))

        # Divide by 2 because the `from_center` creates from the half size.
        shrunk_recep = mn.Range3D.from_center(
            recep.center(),
            (recep.size() / 2.0) * recep_place_shrink_factor,
        )
        pos = np.random.uniform(shrunk_recep.min, shrunk_recep.max)
        set_T = mn.Matrix4.translation(pos)
    else:
        raise ValueError(f"Got unexpected at_entity {at_entity}")

    obj_idx = cast(int, sim_info.search_for_entity(obj))
    abs_obj_id = sim.scene_obj_ids[obj_idx]

    # Get the object id corresponding to this name
    rom = sim.get_rigid_object_manager()
    set_obj = rom.get_object_by_id(abs_obj_id)
    set_obj.transformation = set_T
    set_obj.angular_velocity = mn.Vector3.zero_init()
    set_obj.linear_velocity = mn.Vector3.zero_init()
    sim.internal_step(-1)
    set_obj.angular_velocity = mn.Vector3.zero_init()
    set_obj.linear_velocity = mn.Vector3.zero_init()


def is_articulated_object_at_state(
    art_obj: PddlEntity,
    sim_info: PddlSimInfo,
    target_val: float,
    cmp: str,
    joint_dist_thresh: float = 0.1,
) -> bool:
    """
    Checks if an articulated object matches a joint state condition.

    :param cmp: The comparison to use. Can be "greater", "lesser", or "close".
    """

    if not sim_info.check_type_matches(
        art_obj,
        SimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value,
    ):
        raise ValueError(f"Got unexpected entity {art_obj}")
    marker = cast(
        MarkerInfo,
        sim_info.search_for_entity(
            art_obj,
        ),
    )
    cur_value = marker.get_targ_js()
    if cmp == "greater":
        return cur_value > target_val - joint_dist_thresh
    elif cmp == "lesser":
        return cur_value < target_val + joint_dist_thresh
    elif cmp == "close":
        return abs(cur_value - target_val) < joint_dist_thresh
    else:
        raise ValueError(f"Unrecognized comparison {cmp}")


def set_articulated_object_at_state(
    art_obj: PddlEntity, sim_info: PddlSimInfo, target_val: float
) -> None:
    """
    Sets an articulated object joint state to `target_val`.
    """

    sim = sim_info.sim
    rom = sim.get_rigid_object_manager()

    in_pred = sim_info.get_predicate("in")
    poss_entities = [
        e
        for e in sim_info.all_entities.values()
        if e.expr_type.is_subtype_of(
            sim_info.expr_types[SimulatorObjectType.MOVABLE_ENTITY.value]
        )
    ]

    move_objs = []
    for poss_entity in poss_entities:
        bound_in_pred = in_pred.clone()
        bound_in_pred.set_param_values([poss_entity, art_obj])
        if not bound_in_pred.is_true(sim_info):
            continue
        obj_idx = cast(
            int,
            sim_info.search_for_entity(poss_entity),
        )
        abs_obj_id = sim.scene_obj_ids[obj_idx]
        set_obj = rom.get_object_by_id(abs_obj_id)
        move_objs.append(set_obj)

    marker = cast(
        MarkerInfo,
        sim_info.search_for_entity(
            art_obj,
        ),
    )
    pre_link_pos = marker.link_node.transformation.translation
    marker.set_targ_js(target_val)
    post_link_pos = marker.link_node.transformation.translation

    if art_obj.expr_type.is_subtype_of(sim_info.expr_types[CAB_TYPE]):
        # Also move all objects that were in the drawer
        diff_pos = post_link_pos - pre_link_pos
        for move_obj in move_objs:
            move_obj.translation += diff_pos

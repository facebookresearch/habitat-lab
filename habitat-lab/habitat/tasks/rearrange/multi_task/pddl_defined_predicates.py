from typing import Any, Dict, Optional, cast

import habitat_sim
import magnum as mn
import numpy as np
from habitat.sims.habitat_simulator.sim_utilities import get_ao_global_bb
from habitat.tasks.rearrange.marker_info import MarkerInfo
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    PddlEntity, PddlSimInfo, SimulatorObjectType)
from habitat.tasks.rearrange.utils import (place_agent_at_dist_from_pos,
                                           rearrange_logger)

# TODO: Deprecate these and instead represent them as articulated object entity type.
CAB_TYPE = "cab_type"
FRIDGE_TYPE = "fridge_type"


def is_robot_hold_match(
    robot,
    sim_info,
    hold_state: bool,
    obj=None,
):
    robot_id = cast(
        int,
        sim_info.search_for_entity(robot),
    )
    grasp_mgr = sim_info.sim.get_agent_data(robot_id).grasp_mgr

    assert not (obj is not None and hold_state)

    if obj is not None:
        # Robot must be holding desired object.
        obj_idx = cast(int, sim_info.search_for_entity(obj))
        abs_obj_id = sim_info.sim.scene_obj_ids[obj_idx]
        if grasp_mgr.snap_idx != abs_obj_id:
            return False
    elif hold_state and grasp_mgr.snap_idx != None:
        return False
    return True


def set_robot_holding(robot, sim_info, hold_state: bool, obj=None):
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
        # Swap objects to the desired object.
        obj_idx = cast(int, sim_info.search_for_entity(obj))
        agent_data.grasp_mgr.desnap(True)
        sim.internal_step(-1)
        agent_data.grasp_mgr.snap_to_obj(sim.scene_obj_ids[obj_idx])
        sim.internal_step(-1)


def is_inside(obj, recep, sim_info):
    assert sim_info.check_type_matches(
        obj, SimulatorObjectType.MOVABLE_ENTITY.value
    ), f"Bad type {obj=}"
    assert sim_info.check_type_matches(
        recep, SimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value
    ), f"Bad type {recep=}"
    pass

    entity_pos = sim_info.get_entity_pos(obj)
    check_marker = cast(
        MarkerInfo,
        sim_info.search_for_entity(recep),
    )
    # Hack to see if an object is inside the fridge.
    if sim_info.check_type_matches(recep, "fridge_type"):
        global_bb = get_ao_global_bb(check_marker.ao_parent)
    else:
        bb = check_marker.link_node.cumulative_bb
        global_bb = habitat_sim.geo.get_transformed_bb(
            bb, check_marker.link_node.transformation
        )

    return global_bb.contains(entity_pos)


def set_inside(obj, recep, sim_info):
    raise NotImplemented()


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
    at_entity,
    sim_info,
    dist_thresh: float,
    robot=None,
    filter_colliding_states: bool = True,
    angle_noise: float = 0.0,
    num_spawn_attempts: int = 200,
):
    if robot is None:
        agent_data = sim_info.sim.get_agent_data(None)
    else:
        robot_id = cast(
            int,
            sim_info.search_for_entity(robot),
        )
        agent_data = sim.get_agent_data(robot_id)
    targ_pos = sim_info.get_entity_pos(at_entity)
    sim = sim_info.sim

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


def is_object_at(obj, at_entity, sim_info, dist_thresh):
    entity_pos = sim_info.get_entity_pos(obj)
    if sim_info.check_type_matches(at_entity, SimulatorObjectType.GOAL_ENTITY.value):
        targ_idx = cast(
            int,
            sim_info.search_for_entity(at_entity),
        )
        idxs, pos_targs = sim_info.sim.get_targets()
        targ_pos = pos_targs[list(idxs).index(targ_idx)]

        dist = np.linalg.norm(entity_pos - targ_pos)
        if dist >= sim_info.obj_thresh:
            return False
    elif sim_info.check_type_matches(
        at_entity, SimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value
    ):
        # TODO: Fix this logic to be using sim utilities.
        breakpoint()
        recep = cast(mn.Range3D, sim_info.search_for_entity(at_entity))
        return recep.contains(entity_pos)
    else:
        raise ValueError(f"Got unexpected combination of {obj} and {at_entity}")
    return True


def _place_obj_on_goal(target: PddlEntity, sim_info: PddlSimInfo) -> mn.Matrix4:
    """
    Place an object at a goal position.
    """

    sim = sim_info.sim
    targ_idx = cast(
        int,
        sim_info.search_for_entity(target),
    )
    all_targ_idxs, pos_targs = sim.get_targets()
    targ_pos = pos_targs[list(all_targ_idxs).index(targ_idx)]
    return mn.Matrix4.translation(targ_pos)


def set_object_at(obj, at_entity, sim_info, recep_place_shrink_factor: float = 0.8):
    sim = sim_info.sim

    # The source object must be movable.
    if not sim_info.check_type_matches(obj, SimulatorObjectType.MOVABLE_ENTITY.value):
        raise ValueError(f"Got unexpected obj {obj}")

    elif sim_info.check_type_matches(at_entity, SimulatorObjectType.GOAL_ENTITY.value):
        set_T = _place_obj_on_goal(at_entity, sim_info)
    elif sim_info.check_type_matches(
        at_entity, SimulatorObjectType.STATIC_RECEPTACLE_ENTITY.value
    ):
        # Place object on top of receptacle.
        recep = cast(mn.Range3D, sim_info.search_for_entity(at_entity))
        # TODO: This is actually a receptacle type.
        breakpoint()

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
    art_obj, sim_info, target_val: float, cmp: str, dist_thresh: float = 0.1
):

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
        return cur_value > target_val - dist_thresh
    elif cmp == "less":
        return cur_value < target_val + dist_thresh
    elif cmp == "close":
        return abs(cur_value - target_val) < dist_thresh
    else:
        raise ValueError(f"Unrecognized comparison {cmp}")
    return False


def set_articulated_object_at_state(art_obj, sim_info, target_val: float):
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

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np

import habitat_sim
from habitat.tasks.rearrange.utils import get_aabb, make_render_only
from habitat_sim.physics import MotionType


def add_obj(name, sim):
    if "/BOX_" in name:
        size_parts = name.split("/BOX_")[-1]
        size_parts = size_parts.split("_")
        if len(size_parts) == 1:
            size_x = size_y = size_z = float(size_parts[0])
        else:
            size_x = float(size_parts[0])
            size_y = float(size_parts[1])
            size_z = float(size_parts[2])

        obj_mgr = sim.get_object_template_manager()
        template_handle = obj_mgr.get_template_handles("cube")[0]
        template = obj_mgr.get_template_by_handle(template_handle)
        template.scale = mn.Vector3(size_x, size_y, size_z)
        template.requires_lighting = True
        new_template_handle = obj_mgr.register_template(template, "box_new")
        obj_id = sim.add_object(new_template_handle)
        sim.set_object_motion_type(MotionType.DYNAMIC, obj_id)
        return obj_id

    PROP_FILE_END = ".object_config.json"
    use_name = name + PROP_FILE_END

    obj_id = sim.add_object_by_handle(use_name)
    return obj_id


def place_viz_objs(name_trans, sim, obj_ids):
    viz_obj_ids = []
    for i, (_, assoc_obj_idx, trans) in enumerate(name_trans):
        if len(obj_ids) == 0:
            obj_bb = get_aabb(assoc_obj_idx, sim, False)
            obj_mgr = sim.get_object_template_manager()
            template = obj_mgr.get_template_by_handle(
                obj_mgr.get_template_handles("cubeWireframe")[0]
            )
            template.scale = (
                mn.Vector3(obj_bb.size_x(), obj_bb.size_y(), obj_bb.size_z())
                / 2.0
            )
            new_template_handle = obj_mgr.register_template(
                template, "new_obj_viz"
            )
            viz_obj_id = sim.add_object(new_template_handle)
        else:
            viz_obj_id = obj_ids[i]

        make_render_only(viz_obj_id, sim)
        sim.set_transformation(trans, viz_obj_id)
        viz_obj_ids.append(viz_obj_id)
    return viz_obj_ids


def load_articulated_objs(name_obj_dat, sim, obj_ids, auto_sleep=True):
    """
    Same params as `orp.obj_loaders.load_objs`
    """
    art_obj_ids = []
    for i, (name, obj_dat) in enumerate(name_obj_dat):
        trans = obj_dat[0]
        obj_type = obj_dat[1]
        motion_type = MotionType.DYNAMIC
        if obj_type == -2:
            fixed_base = False
        else:
            fixed_base = True
        if len(obj_ids) == 0:
            ao_mgr = sim.get_articulated_object_manager()
            ao = ao_mgr.add_articulated_object_from_urdf(name, fixed_base)
        else:
            ao = obj_ids[i]
        T = mn.Matrix4(trans)
        ao.transformation = T

        # TODO: Broken in release.
        # if auto_sleep:
        #    ao.can_sleep = True
        ao.motion_type = motion_type
        art_obj_ids.append(ao)
    if len(obj_ids) != 0:
        return obj_ids
    return art_obj_ids


def init_art_objs(idx_and_state, sim):
    for art_obj_idx, art_state in idx_and_state:
        # Need to not sleep so the update actually happens
        prev_sleep = sim.get_articulated_object_sleep(art_obj_idx)

        sim.set_articulated_object_positions(art_obj_idx, np.array(art_state))
        # Default motors for all NONROBOT articulated objects.
        for i in range(len(art_state)):
            jms = habitat_sim.physics.JointMotorSettings(
                0.0,  # position_target
                0.0,  # position_gain
                0.0,  # velocity_target
                0.3,  # velocity_gain
                1.0,  # max_impulse
            )
            sim.update_joint_motor(art_obj_idx, i, jms)
        sim.set_auto_clamp_joint_limits(art_obj_idx, True)
        sim.set_articulated_object_sleep(art_obj_idx, prev_sleep)


def load_objs(name_obj_dat, sim, obj_ids, auto_sleep=True):
    """
    - name_obj_dat: List[(str, List[
        transformation as a 4x4 list of lists of floats,
        int representing the motion type
      ])
    """
    static_obj_ids = []
    for i, (name, obj_dat) in enumerate(name_obj_dat):
        if len(obj_ids) == 0:
            obj_id = add_obj(name, sim)
        else:
            obj_id = obj_ids[i]
        trans = obj_dat[0]
        obj_type = obj_dat[1]

        use_trans = mn.Matrix4(trans)
        sim.set_transformation(use_trans, obj_id)
        sim.set_linear_velocity(mn.Vector3(0, 0, 0), obj_id)
        sim.set_angular_velocity(mn.Vector3(0, 0, 0), obj_id)
        sim.set_object_motion_type(MotionType(obj_type), obj_id)
        static_obj_ids.append(obj_id)
    if len(obj_ids) != 0:
        return obj_ids
    return static_obj_ids

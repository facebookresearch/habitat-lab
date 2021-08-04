def is_stable(sim, watch_ids, is_move_legal_dict, num_sec_sim=1):
    dt = 1 / 30
    num_steps = int((1 / dt) * num_sec_sim)

    static_T = [sim.get_transformation(i) for i in watch_ids]
    ret = True

    obs = []
    for _ in range(num_steps):
        sim.step_world(dt)
        img = sim.get_sensor_observations()["rgb"]
        img = np.flip(img, axis=1)
        obs.append(img)
        is_legal = [
            is_move_legal_dict[other_id](pos=sim.get_translation(other_id))
            for other_id in watch_ids
        ]
        if not all(is_legal):
            # Recover from a bad state.
            set_state(static_T, watch_ids, sim)
            ret = False
            break

    for obj_id in watch_ids:
        sim.set_linear_velocity(mn.Vector3(0, 0, 0), obj_id)
        sim.set_angular_velocity(mn.Vector3(0, 0, 0), obj_id)

    return ret, obs


def get_sampled_obj(
    sim,
    pos_gen,
    pos,
    obj_id,
    existing_obj_ids,
    is_move_legal_dict,
    restrict_bbs,
    should_stabilize,
    is_target,
):
    """
    - is_move_legal_dict: Mapping from obj_idx to lambda which says if position
      is valid.
    """
    timeout_tries = 100
    found = False

    # Keep trying until we get a non-overlapping placement.
    for i in range(timeout_tries):
        new_pos = pos_gen.sample(pos, obj_id)
        if new_pos == "void":
            return new_pos
        sim.set_translation(new_pos, obj_id)
        bb = get_aabb(obj_id, sim, transformed=True)

        if inter_any_bb(bb, restrict_bbs):
            continue
        if is_target:
            closest_nav = sim.pathfinder.snap_point(new_pos)
            dist = dist_2d(closest_nav, new_pos)
            if dist > MAX_OBJ_DIST:
                continue
        found = True
        break

    if not found:
        print(
            "E: Could not get a valid position for %i with %s"
            % (obj_id, str(pos_gen))
        )
        return None

    # also provide proper vertical offset
    if pos_gen.should_add_offset():
        new_pos[1] += (bb.size()[1] / 2.0) + 0.01
    return new_pos


def remove_objs(sim, obj_ids):
    for obj_id in obj_ids:
        sim.remove_object(obj_id)


@attr.s(auto_attribs=True, slots=True)
class ObjDat:
    pos_generator: ObjSampler
    fname: str
    rot: mn.Quaternion
    obj_type: int

    def to_output_spec(self):
        trans = mn.Matrix4.from_(self.rot.to_matrix(), mn.Vector3(self.pos))
        trans = mag_mat_to_list(trans)
        return [self.fname, [trans, self.obj_type]]


def place_articulated_objs(obj_dats, sim, obj_ids=[]):
    art_obj_ids = []
    for i, obj_dat in enumerate(obj_dats):
        if obj_dat.obj_type == -2:
            motion_type = MotionType.DYNAMIC
            fixed_base = False
        else:
            motion_type = MotionType(obj_dat.obj_type)
            fixed_base = True

        if len(obj_ids) == 0:
            obj_id = sim.add_articulated_object_from_urdf(
                obj_dat.fname, fixed_base
            )
        else:
            obj_id = obj_ids[i]

        T = mn.Matrix4.from_(obj_dat.rot.to_matrix(), mn.Vector3(obj_dat.pos))

        sim.set_articulated_object_root_state(obj_id, T)
        sim.set_articulated_object_sleep(obj_id, True)
        sim.set_articulated_object_motion_type(obj_id, motion_type)
        art_obj_ids.append(obj_id)
    if len(obj_ids) != 0:
        return obj_ids
    return art_obj_ids


def place_static_objs(
    obj_dats,
    sim,
    get_place_pos_fn=None,
    on_post_place_fn=None,
    obj_ids=[],
):
    """
    - rotations: (str -> mn.Quaternion) mapping from the name of the object to
      its rotation.
    """
    static_obj_ids = []
    for i, obj_dat in enumerate(obj_dats):
        if len(obj_ids) == 0:
            obj_id = add_obj(obj_dat.fname, sim)
        else:
            obj_id = obj_ids[i]

        sim.set_rotation(obj_dat.rot, obj_id)

        pos = get_place_pos_fn(i, obj_dat.fname, obj_id, static_obj_ids, sim)
        if pos is None:
            # Deallocate all objects
            remove_objs(sim, static_obj_ids)
            # Failed to place the object. We need to start all over.
            return None
        use_motion_type = MotionType(obj_dat.obj_type)
        set_void = False
        if pos == "void":
            set_void = True
            # DO NOT LOAD THIS OBJECT
            sim.remove_object(obj_id)
            continue

        sim.set_translation(mn.Vector3(*pos), obj_id)
        if set_void:
            make_render_only(obj_id, sim)
        else:
            sim.set_object_motion_type(use_motion_type, obj_id)
        sim.set_linear_velocity(mn.Vector3(0, 0, 0), obj_id)
        sim.set_angular_velocity(mn.Vector3(0, 0, 0), obj_id)
        if on_post_place_fn is not None:
            on_post_place_fn(i, obj_dat.fname, pos, obj_id, sim)
        static_obj_ids.append(obj_id)
    if len(obj_ids) != 0:
        return obj_ids
    return static_obj_ids

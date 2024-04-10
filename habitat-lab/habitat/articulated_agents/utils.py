import magnum as mn


def get_articulated_agent_camera_transform_from_cam_info(
    articulated_agent,
    cam_info,
):
    # Get the camera's attached link
    if cam_info.attached_link_id == -1:
        link_trans = articulated_agent.sim_obj.transformation
    elif cam_info.attached_link_id == -2:
        # if isinstance(articulated_agent, KinematicHumanoid):
        rot_offset = articulated_agent.offset_transform_base.inverted()
        link_trans = articulated_agent.base_transformation @ rot_offset
        # else:
        # raise Exception(
        #     f"Did not expect cam_info.attached_link_id to be -2 for {articulated_agent} of type:{type(articulated_agent)}."
        # )
    else:
        link_trans = articulated_agent.sim_obj.get_link_scene_node(
            cam_info.attached_link_id
        ).transformation
    # Get the camera offset transformation
    cam_offset_transform = None
    if cam_info.cam_look_at_pos == None:
        # if cam_info.cam_look_at_pos is None that means the camera's pose-transform
        # is described as a position and orientation in the local coordinate space of the parent link or object
        pos = cam_info.cam_offset_pos
        ori = cam_info.cam_orientation
        Mt = mn.Matrix4.translation(pos)
        Mz = mn.Matrix4.rotation_z(mn.Rad(ori[2]))
        My = mn.Matrix4.rotation_y(mn.Rad(ori[1]))
        Mx = mn.Matrix4.rotation_x(mn.Rad(ori[0]))
        cam_offset_transform = Mt @ Mz @ My @ Mx
    else:
        # if cam_info.look_at_pos is not None then we calculate the camera
        # pose-transform as a look-at transform
        cam_offset_transform = mn.Matrix4.look_at(
            cam_info.cam_offset_pos,
            cam_info.cam_look_at_pos,
            mn.Vector3(0, 1, 0),
        )
    cam_trans = link_trans @ cam_offset_transform @ cam_info.relative_transform
    return cam_trans

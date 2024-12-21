

import magnum as mn
import math

def inverse_transform(pos_a, rot_b, pos_b):
    inv_pos = rot_b.inverted().transform_vector(pos_a - pos_b)    
    return inv_pos


# def spot_arm_ik_helper(link_positions, link_rotations, target_pos):
def spot_arm_ik_helper(target_rel_pos, stickiness_distance_fudge):

    num_dof = 8

    result = [0.0] * num_dof

    # bodies
    # 0 '/World/env_0/Spot/base', 
    # 1 '/World/env_0/Spot/arm0_link_sh0', 
    # 2 '/World/env_0/Spot/arm0_link_sh1', 
    # 3 '/World/env_0/Spot/arm0_link_hr0', 
    # 4 '/World/env_0/Spot/arm0_link_el0', 
    # 5 '/World/env_0/Spot/arm0_link_el1', 
    # 6 '/World/env_0/Spot/arm0_link_wr0', 
    # 7 '/World/env_0/Spot/arm0_link_wr1', 
    # 8 '/World/env_0/Spot/arm0_link_fngr'

    # joints
    # 0 base yaw (+ -> left)
    # 1 base pitch (+ -> down)
    # 2 base twist
    # 3 elbow (+ -> down, can't be negative)
    # 4 elbow twist
    # 5 wrist (+ -> down)
    # 6 wrist twist
    # 7 gripper

    min_dist_to_wrist_target = 0.05

    is_out_of_range = False

    # When this is > 0, ik is less likely to activate. If ik wasn't already active,
    # we set to > 0 to discourage ik from becoming active. This prevents noise/oscillation.

    offset_to_shoulder = mn.Vector3(0.29, 0.0, 0.18)

    # target_rel_pos = inverse_transform(target_pos, link_rotations[0], link_positions[0])
    target_rel_pos -= offset_to_shoulder
    # print(target_rel_pos)

    # dist_to_grasp_point = 0.06
    hand_len = 0.178  # (link_positions[8] - link_positions[7]).length() + dist_to_grasp_point
    forearm_len = 0.410  # (link_positions[6] - link_positions[4]).length()
    upper_arm_len = 0.338  # (link_positions[4] - link_positions[1]).length()
    # print(hand_len, forearm_len, upper_arm_len)

    dist_to_target = target_rel_pos.length()

    target_rel_pos_xy = mn.Vector3(target_rel_pos.x, target_rel_pos.y, 0.0)
    dist_to_target_xy = target_rel_pos_xy.length()

    yaw_angle_to_target = math.atan2(target_rel_pos.y, target_rel_pos.x)

    target_rel_pos_xy_norm = target_rel_pos_xy.normalized()
    # hand_vec is a horizontal offset from the grasp point (in the mouth) to the wrist
    # joint.
    hand_vec = target_rel_pos_xy_norm * hand_len

    # wrist_target_pos is the target position for the wrist, which is offset from the
    # actual ik target pos (by hand_vec).
    wrist_target_pos = target_rel_pos - hand_vec
    dist_to_wrist_target = wrist_target_pos.length()
    dist_to_wrist_target_xy = mn.Vector3(wrist_target_pos.x, wrist_target_pos.y, 0.0).length()

    a = upper_arm_len
    b = forearm_len

    if dist_to_wrist_target_xy < min_dist_to_wrist_target + stickiness_distance_fudge:
        is_out_of_range = True
        pitch_to_wrist_target = 0.0
        h = (a + b) * 0.75
    else:
        upper_plus_forearm_length = forearm_len + upper_arm_len


        pitch_to_wrist_target = math.atan2(wrist_target_pos.z, dist_to_wrist_target_xy)

        h = dist_to_wrist_target

        def is_valid_triangle(a, b, h):
            # Check all three conditions
            # return (a + b > h + stickiness_distance_fudge) and (a + h > b + stickiness_distance_fudge) and (b + h > a + stickiness_distance_fudge)

            # these should already be true since we exited earlier if we're too close
            assert b + h > a
            assert a + h > b

            # Check if we're too far away.
            return a + b > h + stickiness_distance_fudge
        
        if not is_valid_triangle(a, b, h):
            h = (a + b) * 0.4
            is_out_of_range = True

    # Two bone ik. Imagine a triangle, with one side (h) the straight line to the target.
    # The other two sides are the upper arm and forearm.
    ah_angle = math.acos((a**2 + h**2 - b**2) / (2 * a * h))
    ab_angle = math.acos((a**2 + b**2 - h**2) / (2 * a * b))

    # angle at shoulder is the triangle angle ah plus the angle between h and the horizon.
    result[1] = -ah_angle - pitch_to_wrist_target
    # angle at joint is 180 deg - triangle angle
    result[3] = 3.14159 - ab_angle

    # let's do this logic outside this function
    # if not is_out_of_range:
    #     result[7] = -1.67

    # This fudge is because the forearm effectively has a built-in angle (the "crook" 
    # in the arm). We could compute this fudge angle explicitly from the forearm 
    # geometry if desired.
    fudge = 0.15
    result[3] += fudge

    # this logic makes the hand level with the ground
    result[5] = -result[1] - result[3]

    # turn the arm in the ground plane to face the target (yaw)
    result[0] = yaw_angle_to_target

    is_ik_active = not is_out_of_range
    return is_ik_active, result
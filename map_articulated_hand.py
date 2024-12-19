
import magnum as mn
import math


def compute_signed_angle_from_relative_rotation(q1, q2, local_vec, ref_vec):

    vec1 = q1.transform_vector(local_vec)
    vec2 = q2.transform_vector(local_vec)

    angle = math.acos(mn.math.dot(vec1, vec2))

    cross_result = mn.math.cross(vec1, vec2)

    # Compute the signed angle by projecting the rotation axis onto the local axis
    signed_angle = angle * math.copysign(1.0, mn.math.dot(cross_result, ref_vec))

    return float(signed_angle)


def map_articulated_hand_to_metahand_joint_positions(art_hand_positions, art_hand_rotations):


    joint_positions = [0.0] * 16

    # 0 pointer twist
    # 1 thumb rotate
    # 2 ring twist
    # 3 pinky twist
    # 4 pointer base
    # 5 thumb twist
    # 6 ring base
    # 7 pinky base
    # 8 pointer mid
    # 9 thumb base
    # 10 ring mid
    # 11 pinky mid
    # 12 pointer tip
    # 13 thumb tip
    # 14 ring tip
    # 15 pinky tip

    tmp = art_hand_rotations[2].transform_vector(mn.Vector3(0, 0, 1))
    # print(tmp)
    thumb_rotate = -math.asin(tmp.y)
    # print(tmp.y, thumb_rotate)

    joint_positions[1] = thumb_rotate  # 

    joint_positions[5] = 0.7

    thumb_angles = []
    ref_vec = art_hand_rotations[0].transform_vector(mn.Vector3(0, -1, 0))
    local_vec = mn.Vector3(0, 0, 1)
    for i0, i1 in [(1,2), (2,4)]:
        thumb_angles.append(compute_signed_angle_from_relative_rotation(
            art_hand_rotations[i0],
            art_hand_rotations[i1], 
            local_vec,
            ref_vec))
    # print(thumb_angles)

    # tmp = art_hand_rotations[4].transform_vector(mn.Vector3(0, 0, 1))
    # tmp2 = art_hand_rotations[5].transform_vector(mn.Vector3(0, 0, 1))
    # angle = math.acos(mn.math.dot(tmp, tmp2))
    # print(angle)


    joint_positions[9] = thumb_angles[0] + 1.5
    joint_positions[13] = thumb_angles[1]


    pointer_angles = []
    ref_vec = art_hand_rotations[0].transform_vector(mn.Vector3(1, 0, 0))
    local_vec = mn.Vector3(0, 0, 1)
    for i0, i1 in [(6,7), (7,8), (8,9)]:
        pointer_angles.append(compute_signed_angle_from_relative_rotation(
            art_hand_rotations[i0],
            art_hand_rotations[i1], 
            local_vec,
            ref_vec))
        
    joint_positions[4] = pointer_angles[0]
    joint_positions[8] = pointer_angles[1]
    joint_positions[12] = pointer_angles[2]
    


    middle_angles = []
    ref_vec = art_hand_rotations[0].transform_vector(mn.Vector3(1, 0, 0))
    local_vec = mn.Vector3(0, 0, 1)
    for i0, i1 in [(11,12), (12,13), (13, 14)]:
        middle_angles.append(compute_signed_angle_from_relative_rotation(
            art_hand_rotations[i0],
            art_hand_rotations[i1], 
            local_vec,
            ref_vec))
    # print(middle_angles)
        
    joint_positions[6] = middle_angles[0]
    joint_positions[10] = middle_angles[1]
    joint_positions[14] = middle_angles[2]    


    ring_angles = []
    ref_vec = art_hand_rotations[0].transform_vector(mn.Vector3(1, 0, 0))
    local_vec = mn.Vector3(0, 0, 1)
    for i0, i1 in [(16,17), (17,18), (18, 19)]:
        ring_angles.append(compute_signed_angle_from_relative_rotation(
            art_hand_rotations[i0],
            art_hand_rotations[i1], 
            local_vec,
            ref_vec))
    # print(ring_angles)
        
    joint_positions[7] = ring_angles[0]
    joint_positions[11] = ring_angles[1]
    joint_positions[15] = ring_angles[2]   

    return joint_positions
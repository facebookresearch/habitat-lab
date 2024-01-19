# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import random
import sys
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from math import radians
from typing import Any, Dict

# /Applications/Blender.app/Contents/MacOS/Blender
import bpy
import numpy as np
from mathutils import Vector

# Colors from https://colorbrewer2.org/

colors = [(0.2, 0.5, 0.5)]

# state vars
ob = None
counter = 0
links = []
joints = []

# constants
LINK_NAME_FORMAT = "{bone_name}"
JOINT_NAME_FORMAT = "{bone_name}"
JOINT_TYPE = "spherical"
ORIGIN_NODE_FLOAT_PRECISION = 6
ORIGIN_NODE_FORMAT = "{{:.{0}f}} {{:.{0}f}} {{:.{0}f}}".format(
    ORIGIN_NODE_FLOAT_PRECISION
)
ZERO_ORIGIN_NODE = ET.fromstring('<origin xyz="0 0 0" rpy="0 0 0"/>')
INERTIA_NODE_FMT = (
    '<inertia ixx="{}" ixy="{}" ixz="{}" iyy="{}" iyz="{}" izz="{}" />'
)
BASE_LIMIT_NODE_STR = None
multiplier = 1
root_bone = None


####### Export GLTF code ######


def smplx_export_gltf(filepath: str):
    """Given a generate SMPL-X body model, exports it into a GLB file, used to skin the URDF
    Args:
        filepath: the full path where the output glb file will be saved. Should have the glb extension.
    """
    obj = bpy.context.object

    armature_original = obj.parent
    skinned_mesh_original = obj

    # Operate on temporary copy of skinned mesh and armature
    bpy.ops.object.select_all(action="DESELECT")
    skinned_mesh_original.select_set(True)
    armature_original.select_set(True)
    bpy.context.view_layer.objects.active = skinned_mesh_original
    bpy.ops.object.duplicate()
    skinned_mesh = bpy.context.object
    armature = skinned_mesh.parent

    # Apply armature object location to armature root bone and skinned mesh so that armature and skinned mesh are at origin before export
    bpy.context.view_layer.objects.active = armature
    armature_offset = Vector(armature.location)
    armature.location = (0, 0, 0)
    bpy.ops.object.mode_set(mode="EDIT")
    for edit_bone in armature.data.edit_bones:
        if edit_bone.name != "root":
            edit_bone.translate(armature_offset)

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = skinned_mesh
    mesh_location = Vector(skinned_mesh.location)
    skinned_mesh.location = mesh_location + armature_offset
    bpy.ops.object.transform_apply(location=True)

    # Bake and remove shape keys
    pprint("Baking shape and removing shape keys for shape")

    # Create shape mix for current shape
    bpy.ops.object.shape_key_add(from_mix=True)
    num_shape_keys = len(skinned_mesh.data.shape_keys.key_blocks.keys())

    # Remove all shape keys except newly added one
    bpy.context.object.active_shape_key_index = 0
    for _ in range(0, num_shape_keys):
        bpy.ops.object.shape_key_remove(all=False)

    # Model (skeleton and skinned mesh) needs to have rotation of (90, 0, 0) when exporting so that it will have rotation (0, 0, 0) when imported into Unity
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    skinned_mesh.select_set(True)
    skinned_mesh.rotation_euler = (radians(-90), 0, 0)
    bpy.context.view_layer.objects.active = skinned_mesh
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    skinned_mesh.rotation_euler = (radians(90), 0, 0)
    skinned_mesh.select_set(False)

    armature.select_set(True)
    armature.rotation_euler = (radians(-90), 0, 0)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    armature.rotation_euler = (radians(90), 0, 0)

    # Select armature and skinned mesh for export
    skinned_mesh.select_set(True)

    # Rename armature and skinned mesh to not contain Blender copy suffix
    if "female" in skinned_mesh.name:
        gender = "female"
    elif "male" in skinned_mesh.name:
        gender = "male"
    else:
        gender = "neutral"

    target_mesh_name = "SMPLX-mesh-%s" % gender
    target_armature_name = "SMPLX-%s" % gender

    if target_mesh_name in bpy.data.objects:
        bpy.data.objects[target_mesh_name].name = "SMPLX-temp-mesh"
    skinned_mesh.name = target_mesh_name

    if target_armature_name in bpy.data.objects:
        bpy.data.objects[target_armature_name].name = "SMPLX-temp-armature"
    armature.name = target_armature_name

    bpy.ops.export_scene.gltf(
        filepath=filepath,
        use_selection=True,
        export_format="GLB",
        export_texcoords=True,
        export_normals=True,
        export_tangents=False,
        export_materials="EXPORT",
        export_colors=False,
        export_cameras=False,
        export_yup=True,
        export_animations=False,
        export_skins=True,
        export_all_influences=False,
        export_morph=False,
        export_lights=False,
    )

    pprint("Exported: " + filepath)

    # Remove temporary copies of armature and skinned mesh
    bpy.ops.object.select_all(action="DESELECT")
    skinned_mesh.select_set(True)
    armature.select_set(True)
    bpy.ops.object.delete()

    bpy.ops.object.select_all(action="DESELECT")
    skinned_mesh_original.select_set(True)
    bpy.context.view_layer.objects.active = skinned_mesh_original

    if "SMPLX-temp-mesh" in bpy.data.objects:
        bpy.data.objects["SMPLX-temp-mesh"].name = target_mesh_name

    if "SMPLX-temp-armature" in bpy.data.objects:
        bpy.data.objects["SMPLX-temp-armature"].name = target_armature_name

    return {"FINISHED"}


################


#### Build URDF CODE #####
def get_origin_from_matrix(M) -> ET.Element:
    """Get the origin coordinate and rotation from a matrix M.
    Args:
        M: the matrix for which we want to compute the origin coord.
    Returns:
        origin_xml_node: an xml element with the rotation and coordinates of the origin
    """

    global multiplier
    translation = M.to_translation() / multiplier
    euler = M.to_euler()

    origin_xml_node = ET.Element("origin")
    origin_xml_node.set(
        "rpy", ORIGIN_NODE_FORMAT.format(euler.x, euler.y, euler.z)
    )
    origin_xml_node.set(
        "xyz",
        ORIGIN_NODE_FORMAT.format(translation.x, translation.y, translation.z),
    )

    return origin_xml_node


def get_origin_from_bone(bone) -> ET.Element:
    """Get the origin coordinate and rotation for the bone, at the connection of its parent.
    Args:
        bone: the bone for which we want to compute the origin coord.
    Returns:
        origin_xml_node: an xml element with the rotation and coordinates of the origin
    """
    global multiplier
    translation = (
        bone.matrix_local.to_translation()
        - bone.parent.matrix_local.to_translation()
    )

    translation = translation / multiplier

    origin_xml_node = ET.Element("origin")
    origin_xml_node.set("rpy", "0 0 0")
    origin_xml_node.set(
        "xyz",
        ORIGIN_NODE_FORMAT.format(translation.x, translation.y, translation.z),
    )

    return origin_xml_node


def create_bone_link(this_bone) -> ET.Element:
    """Create an xml link given a child bone and populate the joint connecting the child and parent bone.
    Args:
        this_bone: the bone from which we want to create the link.
    Returns:
        xml_link: the xml link with the bone information.
    """
    global counter

    # Get bone properties
    parent_bone = this_bone.parent
    base_joint_name = JOINT_NAME_FORMAT.format(
        counter=counter, bone_name=this_bone.name
    )

    # ------------- Create joint--------------
    joint = ET.Element("joint")
    joint.set("name", base_joint_name)
    joint.set("type", JOINT_TYPE)

    # create parent and child nodes
    parent_xml_node = ET.Element("parent")
    parent_xml_node.set("link", parent_bone.xml_link_name)

    xml_link = ET.Element("link")
    xml_link_name = this_bone.xml_link_name
    xml_link.set("name", xml_link_name)
    links.append(xml_link)

    child_xml_node = ET.Element("child")
    child_xml_node.set("link", xml_link_name)

    joint.append(parent_xml_node)
    joint.append(child_xml_node)

    limit_node = ET.fromstring(BASE_LIMIT_NODE_STR)
    joint.append(limit_node)

    origin_xml_node = get_origin_from_bone(this_bone)

    joint.append(origin_xml_node)
    joints.append(joint)

    ret_link = xml_link

    # this will be used by the next bone to set the correct parent link
    this_bone.xml_link_name = ret_link.get("name")
    return ret_link


# ==========================================


def create_root_bone_link(this_bone) -> ET.Element:
    """Create an xml link given the root bone.
    Args:
        this_bone: the bone from which we want to create the link. It should not have any parent.
    Returns:
        xml_link: the xml link with the root bone information.
    """
    xml_link = ET.Element("link")
    xml_link_name = this_bone.xml_link_name
    xml_link.set("name", xml_link_name)
    links.append(xml_link)

    this_bone.xml_link_name = xml_link_name
    return xml_link


def get_visual_origin(bone) -> ET.Element:
    """Get the origin coordinate and rotation for the bone center.
    Args:
        bone: the bone for which we want to compute the origin coord.
    Returns:
        origin_xml_node: an xml element with the rotation and coordinates of the origin
    """
    global multiplier
    M = bone.matrix_local
    translation = (bone.tail_local - bone.head_local) / (2 * multiplier)
    rotation = M.to_euler()

    origin_xml_node = ET.Element("origin")
    origin_xml_node.set(
        "rpy", ORIGIN_NODE_FORMAT.format(rotation.x, rotation.y, rotation.z)
    )
    origin_xml_node.set(
        "xyz",
        ORIGIN_NODE_FORMAT.format(translation.x, translation.y, translation.z),
    )

    return origin_xml_node


def bone_to_urdf(this_bone):
    """This function extracts the basic properties of the bone and populates
    links and joint lists with the corresponding urdf nodes
    Args:
        this_bone: a humanoid bone. Should have properties such as name, length, and children.
    """

    global counter, multiplier
    this_bone.xml_link_name = LINK_NAME_FORMAT.format(
        counter=counter, bone_name=this_bone.name
    )

    # Create the joint xml node
    if this_bone.parent and this_bone.name != "pelvis":
        this_xml_link = create_bone_link(this_bone)
    else:
        this_xml_link = create_root_bone_link(this_bone)

    this_xml_link.append(
        ET.fromstring(
            INERTIA_NODE_FMT.format(
                this_bone.ixx,
                this_bone.ixy,
                this_bone.ixz,
                this_bone.iyy,
                this_bone.iyz,
                this_bone.izz,
            )
        )
    )
    this_xml_link.append(
        ET.fromstring(
            '<mass value="{:.6f}"/>'.format(this_bone.body_segment_mass)
        )
    )

    # Create the visual node
    this_color = random.choice(colors)
    this_xml_geom = ET.Element("geometry")
    this_xml_box = ET.Element("box")
    padding = min(this_bone.length / (25 * multiplier), 0.02)
    this_xml_box.set(
        "size",
        "{0} {1} {0}".format(
            0.02, this_bone.length / multiplier - padding * 2
        ),
    )
    this_xml_geom.append(this_xml_box)

    this_xml_material = ET.Element("material")
    this_xml_material.set("name", "mat_col_{}".format(this_color))
    this_xml_color = ET.Element("color")
    this_xml_color.set("rgba", "{:.2f} {:.2f} {:.2f} 1.0".format(*this_color))
    this_xml_material.append(this_xml_color)

    this_xml_visual = ET.Element("visual")
    # this_xml_visual.append(ET.fromstring('<origin rpy="0 0 0" xyz="0 {} 0"/>'.format(this_bone.length/2 + padding)))
    this_xml_visual.append(get_visual_origin(this_bone))
    this_xml_visual.append(this_xml_geom)
    this_xml_visual.append(this_xml_material)

    this_xml_link.append(this_xml_visual)

    # Create the collision node
    this_xml_collision = ET.Element("collision")
    this_xml_collision.append(get_visual_origin(this_bone))
    this_xml_collision.append(this_xml_geom)

    this_xml_link.append(this_xml_collision)

    if not this_bone.children:
        pass
        # We reached the end of the chain. Add an end link.
        # create_end_link(this_bone)

    counter += 1


def set_base_limit_str(effort, velocity):
    global BASE_LIMIT_NODE_STR
    BASE_LIMIT_NODE_STR = '<limit effort="{:.4f}" lower="-1.57079632679" upper="1.57079632679" velocity="{:.4f}"/>'.format(
        effort, velocity
    )


def walk_armature(this_bone, handler):
    """Walks recursively through an armature's bones and applies the handler function for each bone.
    Args:
        this_bone: current_root_bone from which the armature should be built
        handler: the function that should be applied to the bone.
    """
    handler(this_bone)
    for child in this_bone.children:
        walk_armature(child, handler)


def smplx_export_urdf(filename: str, settings: Dict[str, Any]):
    """Exports the fbx file in the blender scene into a URDF file, stored in filename.
    Args:
        filename: full path of the urdf file to export.
        settings: a dictionary with export settings, should at least have a field with armature with the armature as a value.
    """
    global LINK_NAME_FORMAT, JOINT_NAME_FORMAT, ob, root_bone, links, joints, multiplier, counter
    counter = 0
    links = []
    joints = []
    if "multiplier" in settings:
        multiplier = settings["multiplier"]
    if "armature" in settings:
        ob = settings["armature"]
    else:
        if "Armature" in bpy.data.objects:
            ob = bpy.data.objects["Armature"]
        else:
            raise Exception("The selected object has no armature")
    # find the root bone, there can be only one
    root_bone = None

    for b in ob.data.bones:
        if not b.parent:
            if root_bone:
                raise Exception("More than one root bone found")
            root_bone = b

    if "link_name_format" in settings:
        LINK_NAME_FORMAT = settings["link_name_format"]

    if "joint_name_format" in settings:
        JOINT_NAME_FORMAT = settings["joint_name_format"]

    effort, velocity = (100, 3)

    if "def_limit_effort" in settings:
        effort = settings["def_limit_effort"]

    if "def_limit_vel" in settings:
        velocity = settings["def_limit_vel"]

    set_base_limit_str(effort, velocity)

    # We want to start at the pelvis
    walk_armature(root_bone.children[0], bone_to_urdf)

    # add all the joints and links to the root
    root_xml = ET.Element("robot")  # create <robot name="test_robot">
    root_xml.set("name", ob.name)

    root_xml.append(ET.Comment("LINKS"))
    for l in links:
        root_xml.append(l)

    root_xml.append(ET.Comment("JOINTS"))
    for j in joints:
        root_xml.append(j)

    # dump the xml string
    ET_raw_string = ET.tostring(root_xml, encoding="unicode")
    dom = minidom.parseString(ET_raw_string)
    ET_pretty_string = dom.toprettyxml()

    with open(filename, "w") as f:
        f.write(ET_pretty_string)

    return ET_pretty_string


#########


def pprint(data):
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == "CONSOLE":
                override = {"window": window, "screen": screen, "area": area}
                bpy.ops.console.scrollback_append(
                    override, text=str(data), type="OUTPUT"
                )


def cleanup():
    """Deletes all objects in the blender scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


class BlenderArgumentParser(argparse.ArgumentParser):
    def parse_args(self):
        parsed_args = []
        if "--" in sys.argv:
            index_val = sys.argv.index("--") + 1
            if index_val < len(sys.argv):
                parsed_args = sys.argv[index_val:]

        return super().parse_args(args=parsed_args)


def set_texture(texture: str):
    """
    Sets a texture to the humanoid, given the file named texture.
    Args:
        texture: path to the texture
    """
    if not os.path.isfile(texture):
        # If the texture file does not exist, it may be part of the blender plugin
        bpy.context.window_manager.smplx_tool.smplx_texture = texture
        bpy.ops.object.smplx_set_texture()
    else:
        # Set the texture manually
        # code adapted from https://github.com/Meshcapade/SMPL_blender_addon/blob/main/meshcapade_addon/operators.py#L426
        obj = bpy.context.object
        if (len(obj.data.materials) == 0) or (obj.data.materials[0] is None):
            print({"WARNING"}, "Selected mesh has no material: %s" % obj.name)

        mat = obj.data.materials[0]
        links = mat.node_tree.links
        nodes = mat.node_tree.nodes

        # Find texture node
        node_texture = None
        for node in nodes:
            if node.type == "TEX_IMAGE":
                node_texture = node
                break
        # Find shader node
        node_shader = None
        for node in nodes:
            if node.type.startswith("BSDF"):
                node_shader = node
                break

        if texture == "NONE":
            # Unlink texture node
            if node_texture is not None:
                for link in node_texture.outputs[0].links:
                    links.remove(link)

                nodes.remove(node_texture)

                # 3D Viewport still shows previous texture when texture link is removed via script.
                # As a workaround we trigger desired viewport update by setting color value.
                node_shader.inputs[0].default_value = node_shader.inputs[
                    0
                ].default_value
        else:
            if node_texture is None:
                node_texture = nodes.new(type="ShaderNodeTexImage")

            if (texture == "UV_GRID") or (texture == "COLOR_GRID"):
                if texture not in bpy.data.images:
                    bpy.ops.image.new(name=texture, generated_type=texture)
                image = bpy.data.images[texture]
            else:
                if texture not in bpy.data.images:
                    texture_path = texture
                    image = bpy.data.images.load(texture_path)
                else:
                    image = bpy.data.images[texture]

            node_texture.image = image

            if len(node_texture.outputs[0].links) == 0:
                links.new(node_texture.outputs[0], node_shader.inputs[0])

        # Switch viewport shading to Material Preview to show texture
        if bpy.context.space_data and bpy.context.space_data.type == "VIEW_3D":
            bpy.context.space_data.shading.type = "MATERIAL"


def setup_bones():
    """Set mass and inertia values to the bones."""
    bpy.types.Bone.xml_link_name = bpy.props.StringProperty(
        name="URDF xml link name", default="unset"
    )
    bpy.types.Bone.body_segment_mass = bpy.props.FloatProperty(
        name="URDF body segment mass", default=5.0
    )
    bpy.types.Bone.ixx = bpy.props.FloatProperty(
        name="Inertia value XX", default=1.0
    )
    bpy.types.Bone.iyy = bpy.props.FloatProperty(
        name="Inertia value YY", default=1.0
    )
    bpy.types.Bone.izz = bpy.props.FloatProperty(
        name="Inertia value ZZ", default=1.0
    )
    bpy.types.Bone.ixy = bpy.props.FloatProperty(
        name="Inertia value XY", default=0.0
    )
    bpy.types.Bone.ixz = bpy.props.FloatProperty(
        name="Inertia value XZ", default=0.0
    )
    bpy.types.Bone.iyz = bpy.props.FloatProperty(
        name="Inertia value YZ", default=0.0
    )
    bpy.types.EditBone.body_segment_mass = bpy.props.FloatProperty(
        name="URDF body segment mass", default=5.0
    )
    bpy.types.EditBone.ixx = bpy.props.FloatProperty(
        name="Inertia value XX", default=1.0
    )
    bpy.types.EditBone.iyy = bpy.props.FloatProperty(
        name="Inertia value YY", default=1.0
    )
    bpy.types.EditBone.izz = bpy.props.FloatProperty(
        name="Inertia value ZZ", default=1.0
    )
    bpy.types.EditBone.ixy = bpy.props.FloatProperty(
        name="Inertia value XY", default=0.0
    )
    bpy.types.EditBone.ixz = bpy.props.FloatProperty(
        name="Inertia value XZ", default=0.0
    )
    bpy.types.EditBone.iyz = bpy.props.FloatProperty(
        name="Inertia value YZ", default=0.0
    )


def main():
    parser = BlenderArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/humanoids/humanoid_data",
        help="Folder where to output body files",
    )
    parser.add_argument(
        "--body-file",
        type=str,
        help="(Optional) File with body parameters, including gender and betas. If not provided, will generate a single shape with random parameters.",
    )

    args = parser.parse_args()

    output_path = args.output_dir
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    if args.body_file is None:
        body_info: list[dict] = [{}]
    else:
        with open(args.body_file, "r") as f:
            body_info = json.load(f)

    fbx_names = []
    export_glb = True
    export_urdf = True
    index_body = 0
    genders = ["neutral", "male", "female"]
    cleanup()
    for index, curr_body_info in enumerate(body_info):
        index_body = index
        cleanup()
        # Set gender
        if "gender" in curr_body_info:
            assert curr_body_info["gender"] in genders
            gender = curr_body_info["gender"]
        else:
            gender = random.choice(genders)

        bpy.context.window_manager.smplx_tool.smplx_gender = gender
        bpy.ops.scene.smplx_add_gender()

        # Set texture
        if "texture" in curr_body_info:
            texture = curr_body_info["texture"]
            set_texture(texture)
        else:
            if gender != "neutral":
                if gender == "female":
                    texture = "smplx_texture_f_alb.png"
                else:
                    texture = "smplx_texture_m_alb.png"
                set_texture(texture)

        obj = bpy.context.object
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.smplx_random_shape()

        if "betas" in curr_body_info:
            betas = curr_body_info["betas"]
            assert len(betas) == 10
        else:
            betas = np.random.rand(10)

        # Add a hand pose
        handpose = "relaxed"
        bpy.context.window_manager.smplx_tool.smplx_hand_pose = handpose

        # Set the shape
        ind = 0

        for key_block in obj.data.shape_keys.key_blocks:
            if ind == 10:
                break
            if key_block.name.startswith("Shape"):
                key_block.value = betas[ind]
                ind += 1

        bpy.ops.object.smplx_snap_ground_plane()

        avatar_name = curr_body_info.get(
            "name", "avatar_{}".format(index_body)
        )
        avatar_dir = "{}/{}".format(output_path, avatar_name)
        if not os.path.isdir(avatar_dir):
            os.mkdir(avatar_dir)
        glb_path = "{}/{}/{}.glb".format(output_path, avatar_name, avatar_name)
        fbx_path = "{}/{}/{}.fbx".format(output_path, avatar_name, avatar_name)
        config_file = "{}/{}/{}.ao_config.json".format(
            output_path, avatar_name, avatar_name
        )

        config_dict = {
            "render_asset": "{}.glb".format(avatar_name),
            "debug_render_primitives": False,
            "semantic_id": 100,
        }
        with open(config_file, "w+") as f:
            f.write(json.dumps(config_dict))

        if export_glb:
            smplx_export_gltf(filepath=glb_path)

        bpy.ops.object.smplx_export_fbx(
            filepath=fbx_path, target_format="UNITY", export_shape_keys="NONE"
        )
        fbx_names.append(fbx_path)

    # Import FBX and export URDF

    if export_urdf:
        for fbx_name in fbx_names:
            cleanup()
            urdf_name = fbx_name.replace(".fbx", ".urdf")
            bpy.ops.import_scene.fbx(
                filepath=fbx_name, automatic_bone_orientation=True
            )
            armature_object = None
            for obj in bpy.context.selected_objects:
                if obj.type == "ARMATURE":
                    armature_object = obj
                    break
            if armature_object == None:
                print("No armature is selected.")
            setup_bones()
            bpy.ops.object.mode_set(mode="OBJECT")
            smplx_export_urdf(urdf_name, {"armature": armature_object})


if __name__ == "__main__":
    # To run this script, use the following command to generate a humanoid avatar with a random shape and gender:
    # For MacOS, the path to blender typically is /Applications/Blender.app/Contents/MacOS/Blender
    # path_to_blender -b -P 'scripts/export_smplx_bodies.py'
    # For more details an options, check out the README:
    # https://github.com/facebookresearch/habitat-lab/tree/main/habitat-lab/habitat/articulated_agents/humanoids#humanoid-design
    main()

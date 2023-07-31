#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random

import bpy
import mathutils

# the generic prefix marking an object as a mesh receptacle
mesh_receptacle_id_string = "receptacle_mesh_"


def get_mesh_area_and_avg_height(mesh_obj):
    """
    Compute and return the area of a mesh object and its average vertex Y value.
    """
    assert mesh_obj.type == "MESH", "Given object is not a mesh."

    # compute the face area
    mesh_area = 0
    avg_y = 0.0
    for face in mesh_obj.data.polygons:
        indices = face.vertices
        assert len(indices) == 3, "Mesh must be triangulated."
        mesh_area += mathutils.geometry.area_tri(
            mesh_obj.data.vertices[indices[0]].co,
            mesh_obj.data.vertices[indices[1]].co,
            mesh_obj.data.vertices[indices[2]].co,
        )
        for index in indices:
            avg_y += mesh_obj.data.vertices[index].co[1]
    avg_y /= len(mesh_obj.data.polygons)
    return mesh_area, avg_y


def cull_floor_height_receptacles(eps: float = 0.05) -> None:
    """
    Deletes receptacle meshes which are likely floor areas.
    1. Compute the area and Y average of all receptacle meshes.
    2. The largest area mesh is assumed to be the floor.
    3. The floor mesh and all other meshes with similar Y avg are deleted.

    :param eps: epsilon threshold for floor receptacle classification
    """
    mesh_receptacles = get_mesh_receptacle_objects()
    mesh_details = {}
    floor_mesh_height = 0
    floor_mesh_area = 0
    for mesh in mesh_receptacles:
        mesh_details[mesh.name] = get_mesh_area_and_avg_height(mesh)
        if mesh_details[mesh.name][0] > floor_mesh_area:
            floor_mesh_area = mesh_details[mesh.name][0]
            floor_mesh_height = mesh_details[mesh.name][1]

    print(f"Floor area {floor_mesh_area} and height {floor_mesh_height}")

    # delete meshes with floor height
    print("Meshes culled for floor height:")
    for mesh_name, details in mesh_details.items():
        if abs(details[1] - floor_mesh_height) < eps:
            print(f"{mesh_name} with height {details[1]} deleted.")
            bpy.data.objects.remove(
                bpy.data.objects[mesh_name], do_unlink=True
            )


def collect_stage_paths(data_dir: str):
    """
    Recursive function to collect paths to all directories with island objs, navmesh, and render asset cache file
    """
    dir_paths = []
    has_navmesh = False
    has_render_cache = False
    has_obj = False
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            # recurse into directories
            dir_paths.extend(collect_stage_paths(item_path))
        elif os.path.isfile(item_path):
            if item.endswith(".navmesh"):
                has_navmesh = True
            elif item.endswith(".obj"):
                has_obj = True
            elif item.endswith("render_asset_path.txt"):
                has_render_cache = True
    if has_navmesh and has_render_cache and has_obj:
        dir_paths.append(data_dir)
    return dir_paths


def get_mesh_receptacle_objects():
    """
    Return a list of all mesh receptacle objects in the scene.
    """
    mesh_receptacles = [
        x
        for x in bpy.data.objects.values()
        if mesh_receptacle_id_string in x.name
    ]
    return mesh_receptacles


def clear_scene():
    """
    Clear the entire scene of all meshes and resources.
    """
    objs = bpy.data.objects
    for objs_name in objs.keys():
        bpy.data.objects.remove(objs[objs_name], do_unlink=True)

    # remove stale data blocks from memory
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


def clear_navmeshes():
    """
    Delete all mesh receptacle objects.
    """
    mesh_receptacles = get_mesh_receptacle_objects()
    for mesh_obj in mesh_receptacles:
        bpy.data.objects.remove(mesh_obj, do_unlink=True)


def load_island_mesh(datapath):
    """
    Load and name a single island mesh component.
    """
    if os.path.isfile(datapath):
        if datapath.endswith(".obj"):
            bpy.ops.import_scene.obj(filepath=datapath)
        elif datapath.endswith(".ply"):
            bpy.ops.import_mesh.ply(filepath=datapath)
        else:
            print(
                f"Cannot process receptacles from this format '{datapath.split('.')[-1]}'. Use .ply or .obj"
            )
            return
        mesh_objects = bpy.context.selected_objects
        for mesh_obj in mesh_objects:
            mesh_obj.name = mesh_receptacle_id_string


def load_island_meshes(datapath):
    """
    Load a set of island objs indexed 0-N from a directory.
    """
    assert os.path.exists(datapath)
    for entry in os.listdir(datapath):
        entry_path = os.path.join(datapath, entry)
        if os.path.isfile(entry_path) and entry.endswith(".obj"):
            load_island_mesh(entry_path)


def load_render_asset_from_cache(render_asset_cache_path):
    assert os.path.isfile(
        render_asset_cache_path
    ), f"'{render_asset_cache_path}' does not exist."
    assert render_asset_cache_path.endswith(
        ".txt"
    ), "must be a txt file containing only the render asset path."
    with open(render_asset_cache_path, "r") as f:
        render_asset_path = f.readline().strip("\n")
        assert os.path.isfile(render_asset_path)
        if render_asset_path.endswith(".glb"):
            bpy.ops.import_scene.gltf(filepath=render_asset_path)
        elif render_asset_path.endswith(".obj"):
            bpy.ops.import_scene.obj(filepath=render_asset_path)
        elif render_asset_path.endswith(".ply"):
            bpy.ops.export_mesh.ply(filepath=render_asset_path)
        else:
            raise AssertionError(
                f"Import of filetype '{render_asset_path}' not supported currently, aborthing scene load."
            )

    objs = bpy.context.selected_objects
    # create an empty frame and parent the object
    bpy.ops.object.empty_add(
        type="ARROWS", align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
    )
    frame = bpy.context.selected_objects[0]
    frame.name = "scene_frame"
    frame.rotation_mode = "QUATERNION"
    for obj in objs:
        if obj.parent == None:
            obj.parent = frame


def assign_random_material_colors_to_rec_meshes():
    """
    Assign random colors to all materials attached to 'mesh_receptacle' objects.
    """
    # get all mesh receptacles
    mesh_receptacles = get_mesh_receptacle_objects()
    for mesh_obj in mesh_receptacles:
        # get all materials attached to this object
        material_slots = mesh_obj.material_slots
        for m in material_slots:
            mat = m.material
            # manipulate the material nodes
            if mat.node_tree is not None:
                for node in mat.node_tree.nodes:
                    # print(f" {node.bl_label}")
                    if node.bl_label == "Principled BSDF":
                        # print(f" {dir(node)}")
                        node.inputs["Base Color"].default_value = (
                            random.random(),
                            random.random(),
                            random.random(),
                            1,
                        )


def get_receptacle_metadata(
    object_name, output_directory, mesh_relative_path=""
):
    """
    Generate a JSON metadata dict for the provided receptacle object.
    """
    assert (
        mesh_receptacle_id_string in object_name
    ), f"Are you sure '{object_name}' is a mesh receptacle?"

    obj = bpy.data.objects[object_name]

    receptacle_info = {
        "name": obj.name,
        # NOTE: default hardcoded values for now
        "position": [0, 0, 0],
        "rotation": [1, 0, 0, 0],
        "scale": [1, 1, 1],
        "up": [0, 1, 0],
        # record the relative filepath to ply files
        "mesh_filepath": mesh_relative_path + object_name + ".ply",
    }

    # write the ply files
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)

    bpy.ops.export_mesh.ply(
        filepath=os.path.join(
            output_directory, receptacle_info["mesh_filepath"]
        ),
        use_selection=True,
        use_ascii=True,
        # don't need extra mesh features
        use_colors=False,
        use_uv_coords=False,
        use_normals=False,
        # convert to habitat-ready coordinate system
        axis_forward="-Z",
        axis_up="Y",
    )

    # TODO: object parented mesh receptacles
    # E.g.
    # receptacle_info["parent_object"] = "kitchen_island"
    # receptacle_info["parent_link"] = obj.parent.name.split("link_")[-1]

    return receptacle_info


def write_receptacle_metadata(output_filename, mesh_relative_path=""):
    """
    Collect and write all receptacle metadata to a JSON file.
    """
    user_defined = {}

    mesh_receptacles = get_mesh_receptacle_objects()

    output_directory = output_filename[: -len(output_filename.split("/")[-1])]
    os.makedirs(output_directory, exist_ok=True)

    for rec_obj in mesh_receptacles:
        user_defined[rec_obj.name] = get_receptacle_metadata(
            rec_obj.name, output_directory, mesh_relative_path
        )

    import json

    with open(output_filename, "w") as f:
        json.dump(user_defined, f, indent=4)


################################################
# main
# NOTE: this should be run through the Blender script window, editing parameters as necessary

# NOTE: This should be the global system path of "output_dir" from "generate_receptacle_navmesh_objs.py"
path_to_receptacle_navmesh_assets = (
    "/Users/andrewszot/Documents/code/p-lang/habitat-lab/data/fp_navmeshes"
)

# define the output directory for meshes and metadata
output_dir = "/Users/andrewszot/Documents/code/p-lang/habitat-lab/data/fp_mesh_receptacle_out/"
# Optionally specify a custom relative path between the metadata and meshes.
# For example, "meshes/" for .ply files in a `meshes` sub-directory relative to the .json
mesh_relative_path = ""

# 1. load the assets
mode = "read"
reload_scene = False  # if True, clear all assets and load the scene assets, otherwise assume we're in the same scene and only reload mesh receptacles
stage_index = 1  # determines which asset will be loaded form the directory
cull_floor_like_receptacles = False  # if true, assume the largest navmesh island is the floor and remove any other islands with the same average height
# 2 do manual annotation
# 3. write the plys and metadata
# mode = "write"

if mode == "read":
    # clear any existing island meshes
    if reload_scene:
        clear_scene()
    clear_navmeshes()

    stage_paths = collect_stage_paths(path_to_receptacle_navmesh_assets)
    print(stage_paths)
    assert (
        len(stage_paths) > stage_index
    ), f"Index {stage_index} out of range. {len(stage_paths)} available."

    # first load the islands and the render asset
    load_island_meshes(stage_paths[stage_index])
    if cull_floor_like_receptacles:
        cull_floor_height_receptacles()
    assign_random_material_colors_to_rec_meshes()

    # load the stage render asset
    if reload_scene:
        load_render_asset_from_cache(
            os.path.join(stage_paths[stage_index], "render_asset_path.txt")
        )
elif mode == "write":
    # write the results
    write_receptacle_metadata(
        output_filename=os.path.join(output_dir, "receptacle_metadata.json"),
        mesh_relative_path=mesh_relative_path,
    )

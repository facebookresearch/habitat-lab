#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os

import bpy
import mathutils

# NOTE: tested with Blender 3.x+
# This script should be run from within Blender script interface.

# NOTE: modify this path to include desired output directory
output_filename = "receptacle_output.json"
# the generic prefix marking an object as an aabb receptacle
mesh_receptacle_id_string = "receptacle_aabb_"

# transformation from Blender to Habitat coordinate system
to_hab = mathutils.Quaternion((1.0, 0.0, 0.0), math.radians(-90.0))
# the JSON config dict to fill
user_defined = {}


def write_object_receptacles():
    """
    Write out all AABB Recetpacle metadata for receptacles attached to an object (e.g. a table).
    Use this to export metadata for .object_config.json and .ao_config.json files.
    """
    for obj in bpy.context.scene.objects:
        if "receptacle_" in obj.name:
            receptacle_info = {
                "name": obj.name,
                # NOTE: hardcoded for now, set this yourself
                "parent_object": "kitchen_island",
                "parent_link": obj.parent.name.split("link_")[-1],
                "position": list(obj.location),
                # NOTE: need half-extents for the final size
                "scale": list(obj.scale * 0.5),
                # NOTE: default hardcoded value for now
                "up": [0, 1, 0],
            }

            # get top level parent
            # top_parent = obj.parent
            # while top_parent.parent is not None:
            #    top_parent = top_parent.parent

            user_defined[obj.name] = receptacle_info


def write_global_receptacles():
    """
    Write out all AABB Recetpacle metadata for receptacles in the global scene space.
    Use this to export metadata for .stage_config.json and .scene_instance.json files.
    """
    for obj in bpy.context.scene.objects:
        if "receptacle_" in obj.name:
            receptacle_info = {"name": obj.name}

            location = obj.location.copy()
            rotation = obj.rotation_quaternion.copy()
            location.rotate(to_hab)
            rotation.rotate(to_hab)

            receptacle_info["position"] = list(location)

            receptacle_info["rotation"] = list(rotation)

            # NOTE: need half-extents for the final size
            receptacle_info["scale"] = list(obj.scale * 0.5)

            # NOTE: default hardcoded value for now
            receptacle_info["up"] = [0, 0, 1]

            user_defined[obj.name] = receptacle_info


# main

# pick your mode:
write_global_receptacles()
# write_object_receptacles()

# write the metadata
output_dir = output_filename[: -len(output_filename.split("/")[-1])]
os.makedirs(output_dir, exist_ok=True)
with open(output_filename, "w") as f:
    json.dump(user_defined, f, indent=4)

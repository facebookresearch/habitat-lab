#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import xml.etree.ElementTree as et
from pathlib import Path


def resolve_relative_path(path: str) -> str:
    """
    Remove './' and '../' from path.
    """
    trailing_slash = "/" if path[0] == "/" else ""
    components = path.split("/")
    output_path: list[str] = []
    for component in components:
        if component == ".":
            continue
        elif component == "..":
            assert (
                len(output_path) > 0
            ), "Relative path escaping out of data folder."
            output_path.pop()
        else:
            output_path.append(component)
    return os.path.join(trailing_slash, *output_path)


def resolve_relative_path_with_wildcard(path: str) -> str:
    """
    Remove trailing wildcards from path, then resolve the path (remove '.' and '..').
    """
    while len(path) > 0:
        if path.endswith("*") or path.endswith("/"):
            path = path[:-1]
        else:
            break
    return resolve_relative_path(path)


def is_file_collada(path: str) -> bool:
    """Returns true if the path is a collada (`.dae`) file."""
    extension = Path(path).suffix.lower()
    return extension == ".dae"


def get_dependencies_urdf(urdf_file_path: str) -> set[str]:
    dependencies: set[str] = set()
    urdf_dir = os.path.dirname(urdf_file_path)
    with open(urdf_file_path, "r") as urdf_file:
        urdf_text = urdf_file.read()
        regex = re.compile('filename="(.*?)"')
        matches = re.findall(regex, urdf_text)
        for match in matches:
            if not match.endswith(".xacro"):
                render_asset_path = os.path.join(urdf_dir, match)
                render_asset_path = resolve_relative_path(render_asset_path)
                dependencies.add(render_asset_path)
    return dependencies


def get_dependencies_obj(obj_file_path: str) -> set[str]:
    dependencies: set[str] = set()
    obj_dir = os.path.dirname(obj_file_path)
    try:
        with open(obj_file_path, "r") as obj_file:
            for line in obj_file:
                # Check for material library files
                if line.startswith("mtllib"):
                    parts = line.split()
                    if len(parts) > 1:
                        mtl_file_path = os.path.join(obj_dir, parts[1])
                        dependencies.add(resolve_relative_path(mtl_file_path))
                # Check for texture files in the .obj file (rare, but possible)
                if (
                    line.startswith("map_Kd")
                    or line.startswith("map_Ka")
                    or line.startswith("map_bump")
                ):
                    parts = line.split()
                    if len(parts) > 1:
                        texture_file_path = os.path.join(obj_dir, parts[1])
                        dependencies.add(
                            resolve_relative_path(texture_file_path)
                        )
    except FileNotFoundError:
        print(f"Error: OBJ file '{obj_file_path}' not found.")

    return dependencies


def get_dependencies_mtl(mtl_file_path: str) -> set[str]:
    dependencies: set[str] = set()
    mtl_dir = os.path.dirname(mtl_file_path)
    try:
        with open(mtl_file_path, "r") as mtl:
            for line in mtl:
                if (
                    line.startswith("map_Kd")
                    or line.startswith("map_Ka")
                    or line.startswith("map_bump")
                ):
                    parts = line.split()
                    if len(parts) > 1:
                        texture_file_path = os.path.join(mtl_dir, parts[1])
                        dependencies.add(
                            resolve_relative_path(texture_file_path)
                        )
    except FileNotFoundError:
        print(f"Warning: Material file '{mtl_file_path}' not found.")
    return dependencies


def get_dependencies_dae(dae_file_path: str) -> set[str]:
    dependencies: set[str] = set()
    dae_dir = os.path.dirname(dae_file_path)
    try:
        tree = et.parse(dae_file_path)
        root = tree.getroot()
        namespaces = {
            "collada": "http://www.collada.org/2005/11/COLLADASchema"
        }
        image_elements = root.findall(
            ".//collada:image/collada:init_from", namespaces
        )
        deps = [image.text for image in image_elements if image.text]
        for dep in deps:
            dep_path = os.path.join(dae_dir, dep)
            dependencies.add(dep_path)
    except et.ParseError as e:
        print(f"Error parsing the COLLADA file '{dae_file_path}': {e}")
    except FileNotFoundError:
        print(f"File not found: {dae_file_path}")
    return dependencies


def get_dependencies(file_path: str, max_depth: int = 10) -> set[str]:
    """
    Recursively find all dependencies for a given asset.

    For instance, `.urdf` files may refer to `.obj` files, which may refer to `.mtl` files, which may refer textures.
    """
    dependencies: set[str] = set()
    if max_depth <= 0:
        print(f"Maximum dependency depth reached for '{file_path}'.")
        return dependencies

    if file_path.lower().endswith(".obj"):
        dependencies = get_dependencies_obj(file_path)
    elif file_path.lower().endswith(".mtl"):
        dependencies = get_dependencies_mtl(file_path)
    elif file_path.lower().endswith(".urdf"):
        dependencies = get_dependencies_urdf(file_path)
    elif file_path.lower().endswith(".dae"):
        dependencies = get_dependencies_dae(file_path)

    for dependency in list(dependencies):
        dependencies.update(get_dependencies(dependency, max_depth - 1))

    return dependencies

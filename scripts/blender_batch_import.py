
import bpy
import os

import bpy, addon_utils

# Robust: enable importers for both OBJ and glTF
for mod in ("io_scene_obj", "io_scene_gltf2"):
    try:
        addon_utils.enable(mod, default_set=True, persistent=False)
    except Exception as e:
        print(f"WARNING: couldn't enable {mod}: {e}")

# Sanity check — raise early if operators still missing
if not hasattr(bpy.ops.import_scene, "obj"):
    raise RuntimeError("OBJ importer unavailable (io_scene_obj not enabled/installed).")
if not hasattr(bpy.ops.import_scene, "gltf"):
    raise RuntimeError("glTF importer unavailable (io_scene_gltf2 not enabled/installed).")

# --- config ---
folder = "/home/eric/projects/habitat-llm/data/mochi_vr_data/source/3bcac30fb6c85be424262960b8909586dc73731b"
spacing = 2.0
axis = "X"   # "X" or "Y"

# --- clear the scene (optional) ---
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# --- gather files ---
exts = (".glb", ".gltf", ".obj")
files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
files.sort()

offset = 0.0
for f in files:
    path = os.path.join(folder, f)
    name = os.path.splitext(f)[0]

    # import depending on extension
    if f.lower().endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=path)
    elif f.lower().endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=path)

    imported = bpy.context.selected_objects

    # group under an empty named after the file
    parent_empty = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(parent_empty)
    for obj in imported:
        obj.parent = parent_empty

    # move the group
    if axis.upper() == "X":
        parent_empty.location.x += offset
    else:
        parent_empty.location.y += offset

    offset += spacing

# optional: save the scene after import
# bpy.ops.wm.save_as_mainfile(filepath="/absolute/path/out.blend")

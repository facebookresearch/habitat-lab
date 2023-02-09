Receptacle Automation Pipeline
==============================

The utilities in this directory are intended to assist users with annotating receptacles for procedural clutter object placement (e.g. for Habitat 2.0 rearrangement tasks).

*A **receptacle** is typically defined as an object or space used to contain something.*

# Context

Many Embodied AI (EAI) tasks (e.g. robotic object rearrangement) involve context rich scenes with a variety of small clutter objects placed in and around larger furniture objects and architectural features. For example, utensils and flatware in kitchen cabinets and drawers.

While artists and users can produce individual arrangements of a scene using standard modeling software, an automated, generative approach is desirable for producing large scale variations (e.g. thousands to millions of variations) for use in training and testing AI models.

[Existing tools in Habitat-lab](https://github.com/facebookresearch/habitat-lab/tree/main/habitat-lab/habitat/datasets/rearrange) depend on the pre-process of annotating receptacle metadata for each and every scene. Currently this process is manual, requiring an artist to place bounding boxes using a modeling software (e.g. Blender) and export a JSON configuration object which is parsed by Habitat sampling logic. See [“The Manual Process”](#the-manual-process) below for details.

## Citation
[Habitat 2.0: Training Home Assistants to Rearrange their Habitat](https://arxiv.org/abs/2106.14405) Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Chaplot, Oleksandr Maksymets, Aaron Gokaslan, Vladimir Vondrus, Sameer Dharur, Franziska Meier, Wojciech Galuba, Angel Chang, Zsolt Kira, Vladlen Koltun, Jitendra Malik, Manolis Savva, Dhruv Batra. Advances in Neural Information Processing Systems (NeurIPS), 2021.

# The Semi-Automated Receptacle Annotation Process
This approach utilizes Habitat-sim’s [Recast|Detour NavMesh](https://aihabitat.org/docs/habitat-sim/habitat_sim.nav.PathFinder.html) integration to compute a set of surfaces which may support or contain the clutter objects. The resulting mesh data is then post-processed into mesh receptacle data structures and manually culled or adjusted by an artist or user in Blender.

The final result is a set of [PLY](https://en.wikipedia.org/wiki/PLY_(file_format)) mesh files and a chunk of JSON metadata which can be included in the stage or object configuration files.

## Setup
First install habitat-sim and habitat-lab with support for Bullet physics as described in the [installation section](https://github.com/facebookresearch/habitat-lab#installation) of Habitat-lab.

- [Download Blender](https://www.blender.org/download/) (tested with v3.3) and install.
  - *Note: run Blender from the terminal on OSX and Linux to see script debug output and errors.*
- Pepare your scene assets in SceneDataset format as described [here](https://aihabitat.org/docs/habitat-sim/attributesJSON.html). For an example, see [ReplicaCAD](https://aihabitat.org/datasets/replica_cad/).
- Configure a custom [NavMeshSettings](https://aihabitat.org/docs/habitat-sim/habitat_sim.nav.NavMeshSettings.html) JSON file or use the provided *clutter_object.navmeshsettings.json* (for small rearrange objects like cans).

## Annotation Process
*NOTE: This process currently supports ONLY global receptacles. While mesh receptacles can be added to object configs and will be parsed by the generator code, this use case has not yet been tested.*

### Overview:
1. [Generate Receptacles:](#1-generate-receptacles) Generate a NavMesh for the scene and export all islands as [.OBJ](https://en.wikipedia.org/wiki/Wavefront_.obj_file) files. (*generate_receptacle_navmesh_obj.py*)
1. [[Blender] Import Receptalce Proposals:](#2-blender-import-receptacle-proposals) Import receptacle meshes in Blender with *blender_receptalce_annotation.py* in "read" mode.
1. [[Blender] Modify Receptacle Set:](#3-blender-modify-receptacle-set)Manually cull, name, and optionally modify the proposed receptacle mesh set.
1. [[Blender] Export Receptacles:](#4-blender-export-receptacles)Export the final metadata JSON and receptacle mesh set as [PLY](https://en.wikipedia.org/wiki/PLY_(file_format)) files with *blender_receptalce_annotation.py* in "write" mode.
1. [Copy Results into SceneDataset:](#5-copy-results-into-scenedataset)Copy the metadata and assets into the SceneDataset directories and files.

### 1. Generate Receptacles:
Generates navmeshes and island .obj files for all stages or scenes in the provided dataset.

#### Basic use:
Generates a default navmesh for a human sized entity and places all output in `navmeshes/` directory:
```bash
#from root habitat-lab/ directory
python scripts/receptacle_annotation/generate_receptacle_navmesh_objs.py --dataset path/to/my_scene.scene_dataset.json
```

optionally provide a modified path for script output:

```bash
--output-dir path/to/directory/
```

#### Custom NavMeshSettings:
You can optionally configure a custom [NavMeshSettings](https://aihabitat.org/docs/habitat-sim/habitat_sim.nav.NavMeshSettings.html) JSON file.

```bash
#from root habitat-lab/ directory
python scripts/receptacle_annotation/generate_receptacle_navmesh_objs.py --dataset path/to/my_scene.scene_dataset.json --navmesh-settings path/to/my_settings.navmesh_settings.json
```

Example *clutter_object.navmeshsettings.json* is provided pre-configured for reasonable receptacle generation results for small clutter objects such as [YCB](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#ycb-benchmarks---object-and-model-set).

See "*Habitat-Sim Basics for Navigation*" from the [ECCV tutorial series](https://aihabitat.org/tutorial/2020/) for more details on configurable navmesh parameters.

### 2. [Blender] Import Receptacle Proposals:
Given 1. the set of navmesh island .objs and 2. the stage asset path from [Generate Receptacles](#1-generate-receptacles), run the *blender_receptacle_annotation.py* script in "read" mode from within Blender.

#### Configure Script Parameters:
Set the path to your output directory from step 1:
```python
path_to_receptacle_navmesh_assets = "navmeshes/"
```
Modify:
- `stage_index` to choose which scene from your dataset to load.
- `reload_scene` to avoid costly asset re-load if iterating on a single scene
- `cull_floor_like_receptacles` to optionally remove any proposed receptacles with average height at floor level.

```python
mode = "read"
reload_scene = True
stage_index = 0 #determines which asset will be loaded from the directory
cull_floor_like_receptacles = False
```

*NOTE: This process will only load the stage asset. Objects added to the scene in scene_instance.json files will not be loaded in Blender automatically.*

After running this script, you should see your stage asset and accompanying island meshes named `receptacle_mesh_`, `receptacle_mesh_.001`, ... `receptacle_mesh_.xxx`.

### 3. [Blender] Modify Receptacle Set:
The goal of this manual phase is to select which meshes will make-up the final receptacle set and choose semantically meaningful names.

*NOTE: all names must begin with prefix 'receptacle_mesh_'.*

#### **Blender UI Tips:**
1. Select an object in the outline and press `numpad-'.'` with cursor in the 3D view panel to center camera view on an object. With the cursor in the outline panel, `numpad-'.'` will center on the object's outline entry.
1. `'TAB'` with an object selected and cursor in the 3D view panel to toggle between `Edit` and `Object` modes.
1. In `Edit` mode you can add, delete, and modify the meshes.
1. If your meshes are not aligned when initially imported, it may be the case that Habitat configs for your scene define a global re-orientation from the base asset. Rotate the parent object of your scene named "scene_frame" to correctly align with the loaded meshes.

#### **Common Operations:**
*NOTE: Any triangle mesh will export correctly. Any new mesh faces MUST be triangulated.*
1. Naming: Default names (e.g. `receptacle_mesh_.001`) are not very informative. Edit text (leaving the `receptacle_mesh_` prefix) to semantically name the receptacles. These will later be used to define receptacle sets for clutter generation.
1. Culling false positives:
Some meshes generated from step 1 will not be reasonable receptacles. Delete these objects in the Blender browser.
1. Modifying meshes: Sometimes a receptacle will have gaps or erronoues faces. Use `Edit` mode to clean up the meshes.
1. Adding new receptacles: Completely new receptacles can be added by duplicating existing meshs or creating new geometry. For example (`Add`->`Mesh`->`Plane`) will create a planar mesh to start with. Remember to triangulate any new geometry before export.

### 4. [Blender] Export Receptacles:
After authoring the desired receptacle mesh set, run the *blender_receptacle_annotation.py* script in "write" mode from within Blender to export the final meshes and metadata.

```python
output_dir = "mesh_receptacle_out/"
mesh_relative_path = ""
mode = "write"
```

All receptacles meshes will be exported as .ply into the configured directory along with *receptacle_metadata.json*. '`mesh_relative_path`' defines the expected relative path between the *.json* and *.ply* files in their final SceneDataset locations.

### 5. Copy Results into SceneDataset:
To use the new annotations in Habitat, you should copy the meshes and metadata into your SceneDataset.

- The contents of *receptacle_metadata.json* can be copied into the `user_defined{}` object of the *.stage_config.json* or *.scene_instance.json* files.
- Meshes should be copied into the scene dataset such that the relative path from the metadata correctly routes to them from the *.json* location.

# The Manual Process
*NOTE: This process currently supports ONLY axis-aligned bounding box (aabb) receptacles.*
1. Load the object or scene in Blender
1. Load the provided metadata export script (*blender_export_aabb_receptacle.py*)
1. Create a new Cube mesh primitive
1. Translate, scale, and rotate the box into the desired position
1. Name the box with prefix "receptacle_aabb_" (e.g. “receptacle_aabb_table_top”, “receptacle_aabb_left_middle_drawer”)
1. Edit the script to choose either "global" or "object" export mode
1. Run an exporter script to produce a JSON
1. Copy JSON into the object or scene’s configuration file under the "*user_defined*" tag.

# Testing Receptacle Annotations:
The easiest way to test your annotations is to run the [rearrange generator](https://github.com/facebookresearch/habitat-lab/tree/main/habitat-lab/habitat/datasets/rearrange) in visual debugging mode with a custom configuration.

1. Direct  `dataset_path:` in *habitat-lab/habitat/datasets/rearrange/configs/all_receptacles_test.yaml* to your SceneDataset config file. Optionally modify object sets, receptacle sets, and sampler paramters.
1. Run the generator `--list` for a quick view of your receptacle configuration:
    ```python
    python habitat-lab/habitat/datasets/rearrange/run_episode_generator.py --list --config habitat-lab/habitat/datasets/rearrange/configs/all_receptacles_test.yaml
    ```
    The output should list all the receptacles you have configured in stage and object config files.
1. Run the generator in verbose debug mode for log output, videos, and images of the sampling process:
    ```python
    python habitat-lab/habitat/datasets/rearrange/run_episode_generator.py --run --debug --verbose --config habitat-lab/habitat/datasets/rearrange/configs/all_receptacles_test.yaml --out mesh_receptacle_out/rearrange_generator_out/ --db-output mesh_receptacle_out/rearrange_generator_out/
    ```
    *NOTE: optionally configure `--out` for generated episodes and `--db-output` for debugging media output.*

    Metrics produced include:
    - process timing (e.g. time to sample N objects)
    - sampling failure statistics
    - dynamic stability analysis: run on after sampling all objects to ensure placements are stable. Issues can indicate poor receptacle support surfaces (e.g. a sloped or un-even bed cover)

    Visual debug output includes:
    - Video showing debug renders of all active receptacles
    - Video from stability test
    - Images of all items sampled
    - Images of all items identified as unstable (prefix "unstable_")

import json
import os
import subprocess
import sys
from typing import List

import pytest
from lxml import etree as ET

repo_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, repo_dir)
isaacsim_dir = repo_dir + "/habitat-lab/habitat/isaac_sim"
sys.path.insert(0, isaacsim_dir)
sys.path.insert(0, repo_dir + "/habitat-lab/habitat/isaac_sim/asset_conversion")

scene_instance_conversion_script = (
    isaacsim_dir + "/asset_conversion/scene_instance_json_to_usd.py"
)
urdf_conversion_script = isaacsim_dir + "/asset_conversion//urdf_to_usd.py"

from scene_instance_json_to_usd import convert_hab_scene
from urdf_to_usd import clean_urdf, convert_urdf

# Initialize the app here
print("Initializing AppLauncher...")

from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app
from pxr import Usd


## Helper functions
def usd_to_habitat_position(position: List[float]) -> List[float]:
    """
    Convert a position from USD (Z-up) to Habitat (Y-up) coordinate system.

    Issac (x, y, z) -> Habitat (-x, z, y)
    """
    x, y, z = position
    return [-x, z, y]


def usd_to_habitat_rotation(rotation: List[float]) -> List[float]:
    """
    Convert a quaternion rotation from USD to Habitat coordinate system.

    Parameters
    ----------
    rotation : list[float]
        Quaternion in USD coordinates [w, x, y, z] (wxyz).

    Returns
    -------
    list[float]
        Quaternion in Habitat coordinates [w, x, y, z] (wxyz).
    """
    HALF_SQRT2 = 0.70710678  # √0.5

    # Combined inverse quaternion transform: (inverse of q_trans)
    # q_x90_inv = [√0.5, √0.5, 0, 0] (wxyz format)
    # q_y180_inv = [0, 0, -1, 0] (wxyz format)
    # q_y90_inv = [HALF_SQRT2, 0.0, HALF_SQRT2, 0.0]

    q_x90_inv = [HALF_SQRT2, HALF_SQRT2, 0.0, 0.0]
    # q_z90_inv = [HALF_SQRT2, 0.0, 0.0, HALF_SQRT2]
    q_y180_inv = [0.0, 0.0, -1.0, 0.0]
    # q_z180_inv = [0.0, 0.0, 0.0, 1.0]

    def quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return [w, x, y, z]

    def quat_inverse(q):
        w, x, y, z = q
        norm = w * w + x * x + y * y + z * z
        return [w / norm, -x / norm, -y / norm, -z / norm]

    # Calculate the inverse of q_x90_inv and q_y180_inv
    q_x90 = quat_inverse(q_x90_inv)
    q_y180 = quat_inverse(q_y180_inv)
    # Calculate q_trans by multiplying q_y180 and q_x90
    q_trans = quat_multiply(q_y180, q_x90)
    # Multiply q_trans with rotation_usd to get the original rotation
    w, x, y, z = rotation
    rotation_hab = quat_multiply(q_trans, rotation)

    return rotation_hab


## scene_instance_json_to_usd.py unit tests


def test_wrong_scene_instance_path():
    scene_filepath = "./NONEXISTENT.scene_instance.json"
    project_root_folder = repo_dir
    scene_usd_filepath = (
        "/home/trandaniel/dev/habitat-lab/data/usd/NONEXISTENT.usda"
    )
    with pytest.raises(FileNotFoundError):
        convert_hab_scene(
            scene_filepath, project_root_folder, scene_usd_filepath
        )


def test_example2_scene_instance():
    scene_filepath = (
        repo_dir
        + "/test/data/usd_conversion_data/EXAMPLE2.scene_instance.json"
    )
    project_root_folder = repo_dir
    scene_usd_filepath = (
        repo_dir + "/data/usd/test_example2_scene_instance.usda"
    )
    object_folder = (
        repo_dir + "/test/data/usd_conversion_data/objects_EXAMPLE2"
    )

    subprocess.run(
        [
            "python",
            scene_instance_conversion_script,
            scene_filepath,
            project_root_folder,
            scene_usd_filepath,
            "--objects_folder",
            object_folder,
        ]
    )

    stage = Usd.Stage.Open(scene_usd_filepath)
    xform_path = "/Scene/OBJECT_1efdc3d37dfab1eb9f99117bb84c59003d684811"
    xform_prim = stage.GetPrimAtPath(xform_path)

    usda_orient_im = list(
        xform_prim.GetAttribute("xformOp:orient").Get().imaginary
    )
    usda_orient_real = xform_prim.GetAttribute("xformOp:orient").Get().real
    # usda_scale = list(xform_prim.GetAttribute("xformOp:scale").Get())
    usda_translate = list(xform_prim.GetAttribute("xformOp:translate").Get())

    # change usd coords back to habitat coords

    usda_translate_hab_coord = usd_to_habitat_position(usda_translate)
    usda_rotation = [usda_orient_real] + usda_orient_im
    usda_rotation_hab_coord = usd_to_habitat_rotation(usda_rotation)

    with open(scene_filepath, "r") as file:
        scene_instance_json_data = json.load(file)

    scene_instance_translation = scene_instance_json_data["object_instances"][
        0
    ]["translation"]
    scene_instance_rotation = scene_instance_json_data["object_instances"][0][
        "rotation"
    ]
    # scene_instance_uniform_scale = scene_instance_json_data[
    #     "object_instances"
    # ][0]["non_uniform_scale"]

    assert usda_translate_hab_coord == pytest.approx(
        scene_instance_translation
    )
    assert usda_rotation_hab_coord == pytest.approx(scene_instance_rotation)
    # TODO: Add Scale to show values are equal, looks like most objects are (1, 1, 1) for the (x, y, z) coordinates in hab space.
    # Not sure, but pxr space is has values from 0 to infinity, and 1 is default, but hab space has
    # -1 values?  Don't know what hab space scaling is.


## urdf_to_usd.py unit tests


def test_clean_urdf():
    urdf_dir = repo_dir + "/test/data/usd_conversion_data/"
    input_file = urdf_dir + "hab_spot_arm_EXAMPLE.urdf"
    output_file = urdf_dir + "hab_spot_arm_clean.urdf"
    correct_file = urdf_dir + "hab_spot_arm_clean_CORRECT.urdf"
    removed_visual = urdf_dir + "hab_spot_arm_clean_visual_removed.urdf"
    removed_visual_CORRECT = (
        urdf_dir + "hab_spot_arm_clean_CORRECT_visual_removed.urdf"
    )

    clean_urdf(input_file, output_file, remove_visual=False)
    clean_urdf(input_file, removed_visual, remove_visual=True)

    tree_test = ET.parse(output_file)
    root_test = tree_test.getroot()
    tree_correct = ET.parse(correct_file)
    root_correct = tree_correct.getroot()

    tree_removed_visual = ET.parse(removed_visual)
    root_removed_visual = tree_removed_visual.getroot()
    tree_removed_visual_CORRECT = ET.parse(removed_visual_CORRECT)
    root_removed_visual_CORRECT = tree_removed_visual_CORRECT.getroot()

    # Update <link> and <joint> names
    for element1, element2 in zip(
        root_test.xpath("//*[@name]"), root_correct.xpath("//*[@name]")
    ):
        assert element1.get("name") == element2.get("name")

    # Update references to <parent link>
    for element1, element2 in zip(
        root_test.xpath("//parent[@link]"),
        root_correct.xpath("//parent[@link]"),
    ):
        assert element1.get("link") == element2.get("link")

    # Update references to <child link>
    for element1, element2 in zip(
        root_test.xpath("//child[@link]"), root_correct.xpath("//child[@link]")
    ):
        assert element1.get("link") == element2.get("link")

    # Optionally remove <visual> elements
    assert len(root_removed_visual.xpath("//visual")) == 0
    assert len(root_removed_visual_CORRECT.xpath("//visual")) == 0
    
    if os.path.exists(output_file):
        os.remove(output_file)
        
    if os.path.exists(removed_visual):
        os.remove(removed_visual)

def test_convert_urdf():
    urdf_dir = repo_dir + "/test/data/usd_conversion_data/"
    clean_urdf_filepath = urdf_dir + "hab_spot_arm_clean_CORRECT.urdf"
    output_usd = urdf_dir + "hab_spot_arm_test_convert_urdf.usda"

    convert_urdf(clean_urdf_filepath, output_usd)
    # TODO: Issac sim drops a prim. "Warning: Prim not found for link: fl_uleg"

    assert True


def test_add_habitat_visual_metadata_for_articulation():
    urdf_dir = repo_dir + "/test/data/usd_conversion_data/"
    converted_clean_usda = urdf_dir + "hab_spot_arm_test_convert_urdf.usda"
    reference_urdf_filepath = urdf_dir + "hab_spot_arm_EXAMPLE.urdf"
    out_usd_filepath = urdf_dir + "hab_spot_arm_with_hab_metadata.usda"
    project_root_folder = repo_dir

    subprocess.run(
        [
            "python",
            urdf_conversion_script,
            "add_habitat_visual_metadata_for_articulation",
            converted_clean_usda,
            reference_urdf_filepath,
            out_usd_filepath,
            project_root_folder,
        ]
    )

    # Parse the URDF file
    urdf_tree = ET.parse(reference_urdf_filepath)
    urdf_root = urdf_tree.getroot()

    # Get the robot name from the URDF
    robot_name = urdf_root.get("name")

    # Build visual metadata dictionary
    visual_metadata = {}
    urdf_dir = os.path.dirname(os.path.abspath(reference_urdf_filepath))

    for link in urdf_root.findall("link"):
        link_name = link.get("name")
        visual = link.find("visual")
        if visual is not None:
            geometry = visual.find("geometry")
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None:
                    filename = mesh.get("filename")
                    asset_path = os.path.relpath(
                        os.path.abspath(os.path.join(urdf_dir, filename)),
                        start=project_root_folder,
                    )
                    scale = (1.0, 1.0, 1.0)  # Default scale

                    # Check for scale in the <mesh> element
                    scale_element = mesh.find("scale")
                    if scale_element is not None:
                        scale = tuple(
                            map(float, scale_element.text.split())
                        )  # type: ignore # noqa: ALL

                    # Replace periods with underscores for USD-safe names
                    # todo: use a standard get_sanitized_usd_name function here
                    safe_link_name = link_name.replace(".", "_")
                    visual_metadata[safe_link_name] = {
                        "assetPath": asset_path,
                        "assetScale": scale,
                    }

    # Open the USD file
    stage_before_metadata = Usd.Stage.Open(converted_clean_usda)
    stage_output_metadata = Usd.Stage.Open(out_usd_filepath)

    # See if habitatVisual data was contained inside usda file
    for link_name, metadata in visual_metadata.items():
        prim_path = f"/{robot_name}/{link_name}"
        prim = stage_before_metadata.GetPrimAtPath(prim_path)

        prim_output_metadata = stage_output_metadata.GetPrimAtPath(prim_path)
        if prim:
            assert (
                metadata["assetPath"]
                == prim_output_metadata.GetAttribute(
                    "habitatVisual:assetPath"
                ).Get()
            )
            assert (
                metadata["assetScale"]
                == prim_output_metadata.GetAttribute(
                    "habitatVisual:assetScale"
                ).Get()
            )

if __name__ == "__main__":
    test_clean_urdf()
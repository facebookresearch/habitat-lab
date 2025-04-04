import json
import torch
import torch.nn.functional as F
import magnum as mn
import numpy as np
import os

def quaternion_to_matrix(quat):
    quat_np = np.array(quat)
    quat_np /= np.linalg.norm(quat_np)           
    # Convert to Magnum Quaternion
    rotation = mn.Quaternion(mn.Vector3(quat_np[0], quat_np[1], quat_np[2]), quat_np[3])
    position=mn.Vector3(1.4,0.88,0.8)
    return mn.Matrix4.from_(rotation.to_matrix(), position)
def generate_json_file(filename=" ", obj_data={}):
    data = {
        "config": None,
        "episodes": []
    }
    
    
    for i, (obj_name, robot_dof) in enumerate(obj_data.items()):
        episode = {
            "episode_id": str(i),
            "scene_id": "fremont_static",
            "scene_dataset_config": "data/usd/scenes/fremont_static_objects.usda",
            "navmesh_path": "data/Fremont-Knuckles/navmeshes/fremont_static.navmesh",
            "additional_obj_config_paths": ["data/usd/objects/dexgraspnet2"],
            "start_position": [1.5, 0.1, -0.6],
            "start_rotation": [0.0, -0.707107, 0.0, 0.707107],
            "info": {
                "object_labels": {"plush2_:0000": "any_targets|0"}
            },
            "ao_states": {},
            "rigid_objs": [
                [
                    obj_name,
                    torch.tensor(
                       robot_dof[1]
                    ).tolist()
                ]
            ],
            "curr_action": "pick",
            "target_joints":torch.tensor(robot_dof[0][0,:]).tolist(),

            "action_target": ["living_room_console",[-5.7, 1.0, -3.8],[-1.04719755,0,0]],
            "targets": {
                "plush2_:0000": torch.tensor([
                    [-0.93324, 0.00000, 0.35926, 2.04542],
                    [0.00000, 1.00000, 0.00000, 0.870047],
                    [-0.35926, 0.00000, -0.93324, 0.75122],
                    [0.00000, 0.00000, 0.00000, 1.00000]
                ]).tolist()
            },
            "markers": [],
            "target_receptacles": [],
            "goal_receptacles": [],
            "name_to_receptacle": {
                "plush2_:0000": "FREMONT-DRESSER_:0000|dresser_top_receptacle_mesh.0000"
            }
        }
        data["episodes"].append(episode)

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"JSON file '{filename}' has been created successfully.")
def acquire_info(case):
    scene = data_cache[case]
    obj_name = case.split('_')[0]
    robot_dof = scene['robot_dof_pos']
    robot_base = scene['robot_root_state']
    robot_xyz = robot_base[:, :3]
    robot_quat = robot_base[:, 3:7]

    obj_state = scene['object_root_state']
    obj_xyz = obj_state[:, :3]
    obj_quat = obj_state[:, 3:7]

    return obj_name, robot_dof, {'xyz': robot_xyz, 'quat':robot_quat}, {'xyz': obj_xyz, 'quat':obj_quat}
if __name__ == "__main__":
    base_dir = "/home/joanne/habitat-lab/dexgraspnet2-large"
    episodes_dir = "/home/joanne/habitat-lab/dexgraspnet2-large/episodes"
    for file_name in os.listdir(base_dir):
        if file_name.endswith(".pth"):
            file_path = os.path.join(base_dir, file_name)
            data = torch.load(file_path, map_location='cpu')

            if 'cache' not in data:
                continue

            data_cache = data['cache']
            cases = list(data_cache.keys())

            # Extract object data
            obj_data = {
                acquire_info(case)[0]: [acquire_info(case)[1], quaternion_to_matrix(acquire_info(case)[2]["quat"][0])]
                for case in cases
            }
            generate_json_file(filename=os.path.join(episodes_dir, f"{os.path.splitext(file_name)[0]}.json"),obj_data=obj_data)

    print(f"All JSON files saved in {episodes_dir}")

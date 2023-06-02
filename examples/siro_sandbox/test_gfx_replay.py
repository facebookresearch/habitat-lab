import gzip
import json
import os

import magnum as mn

import habitat_sim
from habitat_sim.gfx import LightInfo, LightPositionModel
from habitat_sim.utils import viz_utils as vut


def load_json_gzip(filepath):
    with gzip.open(filepath, "rb") as file:
        json_data = file.read().decode("utf-8")
        loaded_data = json.loads(json_data)
    return loaded_data


def make_configuration(settings):
    make_video_during_sim = False
    if "make_video_during_sim" in settings:
        make_video_during_sim = settings["make_video_during_sim"]

    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    # scene setup valid for the following config files:
    # - experiments_hab3/single_agent_pddl_planner_kinematic_oracle_humanoid.yaml
    # - experiments_hab3/pop_play_kinematic_oracle_humanoid_spot.yaml
    backend_cfg.scene_id = (
        "data/replica_cad/configs/scenes/v3_sc2_staging_14.scene_instance.json"
    )
    backend_cfg.scene_dataset_config_file = (
        "data/replica_cad/replicaCAD.scene_dataset_config.json"
    )
    assert os.path.exists(backend_cfg.scene_id)
    backend_cfg.enable_physics = True

    # Enable gfx replay save. See also our call to sim.gfx_replay_manager.save_keyframe()
    # below.
    backend_cfg.enable_gfx_replay_save = True
    backend_cfg.create_renderer = make_video_during_sim

    sensor_cfg = habitat_sim.CameraSensorSpec()
    sensor_cfg.resolution = [544, 720]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def configure_lighting(sim):
    light_setup = [
        LightInfo(
            vector=[1.0, 1.0, 0.0, 1.0],
            color=[18.0, 18.0, 18.0],
            model=LightPositionModel.Global,
        ),
        LightInfo(
            vector=[0.0, -1.0, 0.0, 1.0],
            color=[5.0, 5.0, 5.0],
            model=LightPositionModel.Global,
        ),
        LightInfo(
            vector=[-1.0, 1.0, 1.0, 1.0],
            color=[18.0, 18.0, 18.0],
            model=LightPositionModel.Global,
        ),
    ]
    sim.set_light_setup(light_setup)


replay_filepath_json_gzip = "my_session.0.gfx_replay.json.gz"
replay_filepath_json = replay_filepath_json_gzip[:-3]

replay_keyframes_data = load_json_gzip(replay_filepath_json_gzip)
with open(replay_filepath_json, "w") as f:
    json.dump(replay_keyframes_data, f)

cfg = make_configuration({"make_video_during_sim": True})
sim = habitat_sim.Simulator(cfg)
configure_lighting(sim)

agent_state = habitat_sim.AgentState()
sim.initialize_agent(0, agent_state)

agent_node = sim.get_agent(0).body.object
sensor_node = sim._sensors["rgba_camera"]._sensor_object.object

agent_node.translation = [0.0, 0.0, 0.0]
agent_node.rotation = mn.Quaternion()

player = sim.gfx_replay_manager.read_keyframes_from_file(replay_filepath_json)
assert player

observations = []
print("play replay #0...")
for frame in range(player.get_num_keyframes()):
    player.set_keyframe_index(frame)

    (
        sensor_node.translation,
        sensor_node.rotation,
    ) = player.get_user_transform("user_camera")
    print((sensor_node.translation, sensor_node.rotation))

    observations.append(sim.get_sensor_observations())


vut.make_video(
    observations,
    "rgba_camera",
    "color",
    os.path.basename(replay_filepath_json_gzip),
    open_vid=False,
)

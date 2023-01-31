import os
import sys

import magnum as mn
import numpy as np

import habitat_sim
from habitat_sim.gfx import LightInfo, LightPositionModel
from habitat_sim.utils import gfx_replay_utils
from habitat_sim.utils import viz_utils as vut


def make_configuration():
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    # backend_cfg.scene_id = os.path.join(
    #     # "data/replica_cad/stages/Stage_v3_sc0_staging.glb",
    #     "data/replica_cad/configs/scenes/v3_sc0_staging_00.scene_instance.json",
    # )
    # assert os.path.exists(backend_cfg.scene_id)
    backend_cfg.enable_physics = True
    # backend_cfg.scene_dataset_config_file = (
    #     "data/replica_cad/replicaCAD.scene_dataset_config.json"
    # )
    # backend_cfg.scene_id = "NONE"

    # Enable gfx replay save. See also our call to sim.gfx_replay_manager.save_keyframe()
    # below.
    backend_cfg.enable_gfx_replay_save = True
    backend_cfg.create_renderer = False

    sensor_cfg = habitat_sim.CameraSensorSpec()
    sensor_cfg.resolution = [544, 720]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


cfg = make_configuration()

# use same agents/sensors from earlier, with different backend config
playback_cfg = habitat_sim.Configuration(
    gfx_replay_utils.make_backend_configuration_for_playback(
        need_separate_semantic_scene_graph=False
    ),
    cfg.agents,
)
sim = habitat_sim.Simulator(playback_cfg)


# def configure_lighting(sim):
#     light_setup = [
#         LightInfo(
#             vector=[1.0, 1.0, 0.0, 1.0],
#             color=[18.0, 18.0, 18.0],
#             model=LightPositionModel.Global,
#         ),
#         LightInfo(
#             vector=[0.0, -1.0, 0.0, 1.0],
#             color=[5.0, 5.0, 5.0],
#             model=LightPositionModel.Global,
#         ),
#         LightInfo(
#             vector=[-1.0, 1.0, 1.0, 1.0],
#             color=[18.0, 18.0, 18.0],
#             model=LightPositionModel.Global,
#         ),
#     ]
#     sim.set_light_setup(light_setup)
#
#
# configure_lighting(sim)

agent_state = habitat_sim.AgentState()
sim.initialize_agent(0, agent_state)

agent_node = sim.get_agent(0).body.object
sensor_node = sim._sensors["rgba_camera"]._sensor_object.object

agent_node.translation = [0.0, 0.0, 0.0]
agent_node.rotation = mn.Quaternion()

player = sim.gfx_replay_manager.read_keyframes_from_file(
    # "/Users/andrewszot/Downloads/gala/episode534.replay.json"
    "/srv/share/aszot3/gala/vids/F8b396b7b/episode231.replay.json"
)
observations = []
print("play replay #0...")
for frame in range(player.get_num_keyframes()):
    print("setting keyframe")
    player.set_keyframe_index(frame)
    print("done with keyframe")

    # (
    #     sensor_node.translation,
    #     sensor_node.rotation,
    # ) = player.get_user_transform("sensor")

    # observations.append(sim.get_sensor_observations())

vut.make_video(
    observations,
    "rgba_camera",
    "color",
    "/Users/andrewszot/Downloads/gala/tmp",
    open_vid=False,
)

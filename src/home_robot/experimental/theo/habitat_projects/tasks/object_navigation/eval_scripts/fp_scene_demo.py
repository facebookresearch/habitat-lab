import cv2

import habitat
from habitat_sim.nav import NavMeshSettings

dataset_config_path = (
    "home_robot/experimental/theo/habitat_projects/datasets/scene_datasets/floorplanner/v1/hab-fp.scene_dataset_config.json"
)


def visualize_fp_scenes(scenes):
    for scene in scenes:
        cfg = habitat.get_config()
        cfg.defrost()
        cfg.SIMULATOR.SCENE_DATASET = dataset_config_path
        cfg.SIMULATOR.SCENE = scene
        cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
        cfg.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = True
        cfg.freeze()

        sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

        navmesh_settings = NavMeshSettings()
        navmesh_settings.set_defaults()
        sim.recompute_navmesh(
            sim.pathfinder, navmesh_settings, include_static_objects=True
        )

        pos = sim.pathfinder.get_random_navigable_point()
        # pos = [-6.7892866 ,  0.19999951, -4.6983275 ]
        rot = sim.get_agent_state().rotation
        sim.set_agent_state(pos, rot)

        cv2.imwrite(f"ego_obs_{scene}.png", sim.render()[:, :, ::-1])

        sim.close(destroy=True)


if __name__ == "__main__":
    scenes = ["102344349"]
    visualize_fp_scenes(scenes)

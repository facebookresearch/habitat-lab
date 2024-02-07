# test_hitl_main.py
import magnum
from hydra import compose, initialize

from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import register_hydra_plugins


class AppStateTest(AppState):
    """
    A minimal HITL test app that loads and steps a Habitat environment, with
    a fixed overhead camera.
    """

    def __init__(self, app_service):
        self._app_service = app_service

    def sim_update(self, dt, post_sim_update_dict):
        assert not self._app_service.env.episode_over
        self._app_service.compute_action_and_step_env()

        # set the camera for the main 3D viewport
        post_sim_update_dict["cam_transform"] = magnum.Matrix4.look_at(
            eye=magnum.Vector3(-20, 20, -20),
            target=magnum.Vector3(0, 0, 0),
            up=magnum.Vector3(0, 1, 0),
        )


def main(config) -> None:
    hitl_main(config, lambda app_service: AppStateTest(app_service))


def test_hitl_main():
    register_hydra_plugins()
    with initialize(
        version_base=None, config_path="../../habitat-hitl/habitat_hitl"
    ):
        cfg = compose(config_name="test_cfg")
    main(cfg)
    print("done!")


test_hitl_main()

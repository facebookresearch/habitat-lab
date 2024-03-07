#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Callable, List, Tuple

from omegaconf import DictConfig

from habitat.articulated_agent_controllers.humanoid_rearrange_controller import (
    HumanoidRearrangeController,
)
from habitat.articulated_agents.humanoids.kinematic_humanoid import (
    KinematicHumanoid,
)
from habitat.tasks.rearrange.articulated_agent_manager import (
    ArticulatedAgentData,
)
from habitat.tasks.rearrange.rearrange_grasp_manager import (
    RearrangeGraspManager,
)
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.environment.controllers.gui_controller import (
    GuiHumanoidController,
)


def file_endswith(filepath: str, end_str: str) -> bool:
    """
    Return whether or not the file ends with a string.
    """
    return filepath.endswith(end_str)


def find_files(
    root_dir: str, discriminator: Callable[[str, str], bool], disc_str: str
) -> List[str]:
    """
    Recursively find all filepaths under a root directory satisfying a particular constraint as defined by a discriminator function.

    :param root_dir: The root directory for the recursive search.
    :param discriminator: The discriminator function which takes a filepath and discriminator string and returns a bool.

    :return: The list of all absolute filepaths found satisfying the discriminator.
    """
    filepaths: List[str] = []

    if not os.path.exists(root_dir):
        print(" Directory does not exist: " + str(dir))
        return filepaths

    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        if os.path.isdir(entry_path):
            sub_dir_filepaths = find_files(entry_path, discriminator, disc_str)
            filepaths.extend(sub_dir_filepaths)
        # apply a user-provided discriminator function to cull filepaths
        elif discriminator(entry_path, disc_str):
            filepaths.append(entry_path)
    return filepaths


class AvatarSwitcher:
    """Helper for changing the active humanoid model, along with its animations and controller."""

    def __init__(
        self,
        app_service: AppService,
        gui_agent_ctrl: GuiHumanoidController,
    ):
        self._app_service: AppService = app_service
        self._gui_agent_ctrl: GuiHumanoidController = gui_agent_ctrl
        self._agent_idx: int = self._gui_agent_ctrl._agent_idx

        self._humanoid_models: List[
            Tuple[str, str]
        ] = self._get_humanoid_models("data/humanoids/humanoid_data/")

        self._avatar_model_idx = 0

    def _get_humanoid_models(self, filepath: str) -> List[Tuple[str, str]]:
        """
        Get a list of humanoid urdfs and accompanying motion files from a directory.
        Assumes naming format between model and motion files: ("female_0.urdf" -> "female_0_motion_data_smplx.pkl")
        """
        models = []
        urdfs = find_files(filepath, file_endswith, ".urdf")
        for urdf in urdfs:
            model_name = urdf.split(".urdf")[0]
            motion_path = model_name + "_motion_data_smplx.pkl"
            if os.path.isfile(motion_path):
                models.append((urdf, motion_path))
        return models

    def _get_next_model(self) -> Tuple[str, str]:
        model = self._humanoid_models[self._avatar_model_idx]
        self._avatar_model_idx = (self._avatar_model_idx + 1) % len(
            self._humanoid_models
        )
        return model

    def switch_avatar(self):
        """Switch the avatar to a new model."""
        sim = self._app_service.sim
        config = self._app_service.config

        model = self._get_next_model()

        # record current humanoid state
        humanoid_agent_data: ArticulatedAgentData = sim.agents_mgr[
            self._agent_idx
        ]
        humanoid = humanoid_agent_data.articulated_agent
        humanoid_controller = self._gui_agent_ctrl._humanoid_controller

        prev_humanoid_base_state = (
            humanoid.base_pos,
            float(humanoid.base_rot),
            humanoid_controller.prev_orientation,
            humanoid_controller.walk_mocap_frame,
        )

        sim.get_articulated_object_manager().remove_object_by_handle(
            humanoid.sim_obj.handle
        )

        # create new humanoid
        agent_config = DictConfig(
            {
                "articulated_agent_urdf": model[0],
                "motion_data_path": model[1],
            }
        )
        humanoid = KinematicHumanoid(agent_config, sim=sim)
        humanoid.reconfigure()

        humanoid.base_pos = prev_humanoid_base_state[0]
        humanoid.base_rot = prev_humanoid_base_state[1]

        # create new grasp manager
        grasp_mgr = RearrangeGraspManager(
            sim, config.habitat.simulator, humanoid, 0
        )

        # maybe re-snap object
        assert len(humanoid_agent_data.grasp_mgrs) == 1
        if humanoid_agent_data.grasp_mgrs[0].snap_idx is not None:
            grasped_obj_id = humanoid_agent_data.grasp_mgrs[0].snap_idx
            grasp_mgr.snap_to_obj(grasped_obj_id)

        humanoid_agent_data.articulated_agent = humanoid
        humanoid_agent_data.grasp_mgrs = [grasp_mgr]

        # create new humanoid controller
        humanoid_controller = HumanoidRearrangeController(
            walk_pose_path=model[1]
        )
        # todo: support for multiple gui-controlled agents here
        assert len(self._app_service.hitl_config.gui_controlled_agents) == 1
        gui_controlled_agent_config = (
            self._app_service.hitl_config.gui_controlled_agents[0]
        )
        humanoid_controller.set_framerate_for_linspeed(
            gui_controlled_agent_config.lin_speed,
            gui_controlled_agent_config.ang_speed,
            sim.ctrl_freq,
        )

        humanoid_controller.reset(humanoid.base_transformation)

        humanoid_controller.prev_orientation = prev_humanoid_base_state[2]
        humanoid_controller.walk_mocap_frame = prev_humanoid_base_state[3]

        self._gui_agent_ctrl.set_humanoid_controller(humanoid_controller)

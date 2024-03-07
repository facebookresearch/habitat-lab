#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    HumanoidJointActionConfig,
    ThirdRGBSensorConfig,
)


def update_config(
    config,
    show_debug_third_person=False,
    debug_third_person_width=None,
    debug_third_person_height=None,
):
    with habitat.config.read_write(config):  # type: ignore
        habitat_config = config.habitat
        sim_config = habitat_config.simulator
        task_config = habitat_config.task
        gym_obs_keys = habitat_config.gym.obs_keys
        hitl_config = config.habitat_hitl

        agent_config = get_agent_config(sim_config=sim_config)

        if show_debug_third_person:
            sim_config.debug_render = True
            agent_config.sim_sensors.update(
                {
                    "third_rgb_sensor": ThirdRGBSensorConfig(
                        height=debug_third_person_height,
                        width=debug_third_person_width,
                    )
                }
            )
            agent_key = "" if len(sim_config.agents) == 1 else "agent_0_"
            agent_sensor_name = f"{agent_key}third_rgb"
            hitl_config.debug_images.append(agent_sensor_name)
            gym_obs_keys.append(agent_sensor_name)

        for (
            gui_controlled_agent_config
        ) in config.habitat_hitl.gui_controlled_agents:
            gui_controlled_agent_index = (
                gui_controlled_agent_config.agent_index
            )
            # make sure gui_controlled_agent_index is valid
            if not (
                gui_controlled_agent_index >= 0
                and gui_controlled_agent_index < len(sim_config.agents)
            ):
                print(
                    f"habitat_hitl.gui_controlled_agents[i].agent_index ({gui_controlled_agent_index}) "
                    f"must be >= 0 and < number of agents ({len(sim_config.agents)})"
                )
                exit()

            # make sure chosen articulated_agent_type is supported
            gui_agent_key = sim_config.agents_order[gui_controlled_agent_index]
            agent_type = sim_config.agents[
                gui_agent_key
            ].articulated_agent_type
            if agent_type != "KinematicHumanoid" and agent_type != "SpotRobot":
                raise ValueError(
                    f"Selected agent for GUI control is of type {sim_config.agents[gui_agent_key].articulated_agent_type}, "
                    "but only KinematicHumanoid and SpotRobot are supported at the moment."
                )

            # avoid camera sensors for GUI-controlled agents
            gui_controlled_agent_config = get_agent_config(
                sim_config, agent_id=gui_controlled_agent_index
            )
            gui_controlled_agent_config.sim_sensors.clear()

            lab_sensor_names = ["has_finished_oracle_nav"]
            for lab_sensor_name in lab_sensor_names:
                sensor_name = (
                    lab_sensor_name
                    if len(sim_config.agents) == 1
                    else (f"{gui_agent_key}_{lab_sensor_name}")
                )
                if sensor_name in task_config.lab_sensors:
                    task_config.lab_sensors.pop(sensor_name)

            task_measurement_names = [
                "does_want_terminate",
                "bad_called_terminate",
            ]
            for task_measurement_name in task_measurement_names:
                measurement_name = (
                    task_measurement_name
                    if len(sim_config.agents) == 1
                    else (f"{gui_agent_key}_{task_measurement_name}")
                )
                if measurement_name in task_config.measurements:
                    task_config.measurements.pop(measurement_name)

            # todo: decide whether to fix up config here versus validate config
            sim_sensor_names = [
                "head_depth",
                "head_rgb",
                "articulated_agent_arm_depth",
            ]
            for sensor_name in sim_sensor_names + lab_sensor_names:
                sensor_name = (
                    sensor_name
                    if len(sim_config.agents) == 1
                    else (f"{gui_agent_key}_{sensor_name}")
                )
                if sensor_name in gym_obs_keys:
                    gym_obs_keys.remove(sensor_name)

            if agent_type == "KinematicHumanoid":
                # use humanoidjoint_action for GUI-controlled KinematicHumanoid
                # for example, humanoid oracle-planner-based policy uses following actions:
                # base_velocity, rearrange_stop, pddl_apply_action, oracle_nav_action
                task_actions = task_config.actions
                action_prefix = (
                    "" if len(sim_config.agents) == 1 else f"{gui_agent_key}_"
                )
                gui_agent_actions = [
                    action_key
                    for action_key in task_actions.keys()
                    if action_key.startswith(action_prefix)
                ]
                for action_key in gui_agent_actions:
                    task_actions.pop(action_key)

                task_actions[
                    f"{action_prefix}humanoidjoint_action"
                ] = HumanoidJointActionConfig()

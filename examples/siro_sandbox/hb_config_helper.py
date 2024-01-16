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


def update_config_and_args(
    config,
    args,
    show_debug_third_person=False,
    debug_third_person_width=None,
    debug_third_person_height=None,
):
    with habitat.config.read_write(config):  # type: ignore
        habitat_config = config.habitat
        env_config = habitat_config.environment
        sim_config = habitat_config.simulator
        task_config = habitat_config.task
        gym_obs_keys = habitat_config.gym.obs_keys

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
            args.debug_images.append(agent_sensor_name)
            gym_obs_keys.append(agent_sensor_name)

        # Code below is ported from interactive_play.py. I'm not sure what it is for.
        if True:
            if "pddl_success" in task_config.measurements:
                task_config.measurements.pddl_success.must_call_stop = False
            if "rearrange_nav_to_obj_success" in task_config.measurements:
                task_config.measurements.rearrange_nav_to_obj_success.must_call_stop = (
                    False
                )
            if "force_terminate" in task_config.measurements:
                task_config.measurements.force_terminate.max_accum_force = -1.0
                task_config.measurements.force_terminate.max_instant_force = (
                    -1.0
                )

        if args.never_end:
            env_config.max_episode_steps = 0

        if not args.disable_inverse_kinematics:
            if "arm_action" not in task_config.actions:
                raise ValueError(
                    "Action space does not have any arm control so cannot add inverse kinematics. Specify the `--disable-inverse-kinematics` option"
                )
            sim_config.agents.main_agent.ik_arm_urdf = (
                "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
            )
            task_config.actions.arm_action.arm_controller = "ArmEEAction"

        if args.gui_controlled_agent_index is not None:
            # make sure gui_controlled_agent_index is valid
            if not (
                args.gui_controlled_agent_index >= 0
                and args.gui_controlled_agent_index < len(sim_config.agents)
            ):
                print(
                    f"--gui-controlled-agent-index argument value ({args.gui_controlled_agent_index}) "
                    f"must be >= 0 and < number of agents ({len(sim_config.agents)})"
                )
                exit()

            # make sure chosen articulated_agent_type is supported
            gui_agent_key = sim_config.agents_order[
                args.gui_controlled_agent_index
            ]
            if (
                sim_config.agents[gui_agent_key].articulated_agent_type
                != "KinematicHumanoid"
            ):
                print(
                    f"Selected agent for GUI control is of type {sim_config.agents[gui_agent_key].articulated_agent_type}, "
                    "but only KinematicHumanoid is supported at the moment."
                )
                exit()

            # avoid camera sensors for GUI-controlled agents
            gui_controlled_agent_config = get_agent_config(
                sim_config, agent_id=args.gui_controlled_agent_index
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

            sim_sensor_names = ["head_depth", "head_rgb"]
            for sensor_name in sim_sensor_names + lab_sensor_names:
                sensor_name = (
                    sensor_name
                    if len(sim_config.agents) == 1
                    else (f"{gui_agent_key}_{sensor_name}")
                )
                if sensor_name in gym_obs_keys:
                    gym_obs_keys.remove(sensor_name)

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

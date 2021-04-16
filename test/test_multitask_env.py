import os

import pytest

from habitat.config import get_config
from habitat.core.env import MultiTaskEnv
from habitat.tasks.nav.nav import TeleportAction

TEST_CFG_PATH = "configs/test/habitat_multitask_example.yaml"


def get_test_config(name: str):
    # use test dataset for lighter testing
    datapath = (
        "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz"
    )
    cfg = get_config(name)
    if not os.path.exists(cfg.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    if len(cfg.TASKS) < 2:
        pytest.skip(
            "Please use a configuration with at least 2 tasks for testing."
        )
    cfg.defrost()
    cfg.DATASET.DATA_PATH = datapath
    cfg.DATASET.SPLIT = "test"
    # also make sure tasks config are overriden
    for task in cfg.TASKS:
        task.DATASET.DATA_PATH = datapath
        task.DATASET.SPLIT = "test"
    # and work with small observations for testing
    if "RGB_SENSOR" in cfg:
        cfg.RGB_SENSOR.WIDTH = 64
        cfg.RGB_SENSOR.HEIGHT = 64
    if "DEPTH_SENSOR" in cfg:
        cfg.DEPTH_SENSOR.WIDTH = 64
        cfg.DEPTH_SENSOR.HEIGHT = 64
    cfg.freeze()
    return cfg


def test_standard_config_compatibility():
    cfg = get_config("configs/tasks/pointnav.yaml")
    with MultiTaskEnv(config=cfg) as env:
        env.reset()
        actions = 0

        while not env.episode_over:
            # execute random action
            env.step(env.action_space.sample())
            actions += 1
        assert (
            actions >= 1
        ), "You should have performed at least one step with no interruptions"


def test_simple_fixed_change_task():
    cfg = get_test_config(TEST_CFG_PATH)
    cfg.defrost()
    cfg.CHANGE_TASK_BEHAVIOUR.TYPE = "FIXED"
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_EPISODES = 1
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_CUM_STEPS = None
    cfg.CHANGE_TASK_BEHAVIOUR.LOOP = "ORDER"
    # test meant for 2 tasks
    cfg.TASKS = cfg.TASKS[:2]
    cfg.freeze()
    with MultiTaskEnv(config=cfg) as env:
        # we have two tasks, we'll only need task index 0 and 1
        task_idx = 0
        for _ in range(5):
            env.reset()
            assert (
                env._curr_task_idx == task_idx
            ), "Task should change at each new episode"
            task_idx = (task_idx + 1) % 2
            while not env.episode_over:
                # execute random action
                env.step(env.action_space.sample())


def test_fixed_change_multiple_tasks():
    change_after = 3
    cfg = get_test_config(TEST_CFG_PATH)
    cfg.defrost()
    cfg.CHANGE_TASK_BEHAVIOUR.TYPE = "FIXED"
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_EPISODES = change_after
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_CUM_STEPS = None
    cfg.CHANGE_TASK_BEHAVIOUR.LOOP = "ORDER"
    cfg.freeze()
    with MultiTaskEnv(config=cfg) as env:
        task_idx = 0
        for i in range(10):
            env.reset()
            if i > 0 and i % change_after == 0:
                task_idx = (task_idx + 1) % len(cfg.TASKS)
            assert (
                env._curr_task_idx == task_idx
            ), "Task should change every {} episodes".format(change_after)
            while not env.episode_over:
                # execute random action
                env.step(env.action_space.sample())


def test_cum_steps_change_tasks_same_scene():
    change_after = 5
    cfg = get_test_config(TEST_CFG_PATH)
    cfg.defrost()
    cfg.CHANGE_TASK_BEHAVIOUR.TYPE = "FIXED"
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_EPISODES = None
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_CUM_STEPS = change_after
    cfg.CHANGE_TASK_BEHAVIOUR.LOOP = "ORDER"
    cfg.freeze()
    with MultiTaskEnv(config=cfg) as env:
        task_idx = 0
        actions = 0
        for _ in range(10):
            env.reset()

            while not env.episode_over:
                # execute random action
                env.step(env.action_space.sample())
                actions += 1
                if actions >= change_after:
                    actions = 0
                    task_idx = (task_idx + 1) % len(cfg.TASKS)
                assert (
                    env._curr_task_idx == task_idx
                ), "Task should change every {} steps".format(change_after)


def test_cum_steps_change_tasks_different_scene():
    change_after = 3
    cfg = get_test_config(TEST_CFG_PATH)
    cfg.defrost()
    cfg.CHANGE_TASK_BEHAVIOUR.TYPE = "FIXED"
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_EPISODES = None
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_CUM_STEPS = change_after
    cfg.CHANGE_TASK_BEHAVIOUR.LOOP = "ORDER"
    cfg.TASKS[0].DATASET.SPLIT = "train"  # get different split for this task
    cfg.freeze()
    with MultiTaskEnv(config=cfg) as env:
        task_idx = 0
        actions = 0
        for _ in range(10):
            env.reset()

            while not env.episode_over:
                # execute random action
                env.step(env.action_space.sample())
                actions += 1
                if actions >= change_after:
                    actions = 0
                    task_idx = (task_idx + 1) % len(cfg.TASKS)
                assert (
                    env._curr_task_idx == task_idx
                ), "Task should change every {} steps".format(change_after)


def test_ep_or_steps_change_tasks():
    change_after_eps = 2
    change_after_steps = 10
    cfg = get_test_config(TEST_CFG_PATH)
    cfg.defrost()
    cfg.CHANGE_TASK_BEHAVIOUR.TYPE = "FIXED"
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_EPISODES = change_after_eps
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_CUM_STEPS = change_after_steps
    cfg.CHANGE_TASK_BEHAVIOUR.LOOP = "ORDER"
    cfg.freeze()
    with MultiTaskEnv(config=cfg) as env:
        task_idx = 0
        actions = 0
        for i in range(10):
            env.reset()
            if i > 0 and i % change_after_eps == 0:
                task_idx = (task_idx + 1) % len(cfg.TASKS)
            assert (
                env._curr_task_idx == task_idx
            ), "Task should change every {} episodes".format(change_after_eps)
            while not env.episode_over:
                # execute random action
                env.step(env.action_space.sample())
                actions += 1
                if actions >= change_after_steps:
                    actions = 0
                    task_idx = (task_idx + 1) % len(cfg.TASKS)
                assert (
                    env._curr_task_idx == task_idx
                ), "Task should change every {} steps".format(
                    change_after_steps
                )


def test_random_change_tasks():
    change_after = 3
    cfg = get_test_config(TEST_CFG_PATH)
    cfg.defrost()
    cfg.CHANGE_TASK_BEHAVIOUR.TYPE = "RANDOM"
    cfg.CHANGE_TASK_BEHAVIOUR.CHANGE_TASK_PROB = 1.0
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_EPISODES = change_after
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_CUM_STEPS = None
    cfg.CHANGE_TASK_BEHAVIOUR.LOOP = "ORDER"
    cfg.freeze()
    with MultiTaskEnv(config=cfg) as env:
        task_idx = 0
        for i in range(change_after * 2 + 1):
            env.reset()
            if i > 0 and i % change_after == 0:
                task_idx = (task_idx + 1) % len(cfg.TASKS)
            assert (
                env._curr_task_idx == task_idx
            ), "Task should change every {} episodes".format(change_after)

            while not env.episode_over:
                # execute random action
                env.step(env.action_space.sample())
    # it should never change now
    cfg.defrost()
    cfg.CHANGE_TASK_BEHAVIOUR.CHANGE_TASK_PROB = 0.0
    cfg.freeze()
    with MultiTaskEnv(config=cfg) as env:
        task_idx = 0
        for _ in range(change_after * 2 + 1):
            env.reset()
            assert env._curr_task_idx == task_idx, "Task should not change"

            while not env.episode_over:
                # execute random action
                env.step(env.action_space.sample())


def test_random_task_loop():
    change_after = 3
    cfg = get_test_config(TEST_CFG_PATH)
    cfg.defrost()
    cfg.CHANGE_TASK_BEHAVIOUR.TYPE = "FIXED"
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_EPISODES = change_after
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_CUM_STEPS = None
    cfg.CHANGE_TASK_BEHAVIOUR.LOOP = "RANDOM"
    cfg.freeze()
    with MultiTaskEnv(config=cfg) as env:
        task_idx = 0
        for i in range(10):
            env.reset()
            if i > 0 and i % change_after == 0:
                assert (
                    env._curr_task_idx != task_idx
                ), "Task should change every {} episodes".format(change_after)
                task_idx = env._curr_task_idx
            while not env.episode_over:
                # execute random action
                env.step(env.action_space.sample())


def test_different_actions_spaces():
    cfg = get_test_config(TEST_CFG_PATH)
    if len(cfg.TASKS) <= 2:
        pytest.skip("Need 3 tasks to run this test")

    cfg.defrost()
    if "TELEPORT" in cfg.TASKS[0].POSSIBLE_ACTIONS:
        cfg.TASKS[0].POSSIBLE_ACTIONS.remove("TELEPORT")

    cfg.TASKS[1].POSSIBLE_ACTIONS = cfg.TASKS[0].POSSIBLE_ACTIONS + [
        "TELEPORT"
    ]
    cfg.TASKS[2].POSSIBLE_ACTIONS = ["TELEPORT"]
    # change after 1 episode
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_EPISODES = 1
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_CUM_STEPS = None
    cfg.CHANGE_TASK_BEHAVIOUR.LOOP = "FIXED"
    cfg.freeze()
    with MultiTaskEnv(config=cfg) as env:
        for _ in range(10):
            env.reset()
            if env._curr_task_idx == 0:
                assert not env.action_space.contains(
                    TeleportAction(config=cfg, sim=env.sim)
                )
            elif env._curr_task_idx == 1:
                assert env.action_space.contains(
                    TeleportAction(config=cfg, sim=env.sim)
                )
            elif env._curr_task_idx == 2:
                assert (
                    env.action_space
                    == TeleportAction(config=cfg, sim=env.sim).action_space
                )

            while not env.episode_over:
                # execute random action
                action = env.action_space.sample()
                if env._curr_task_idx == 2:
                    # inputting any other action will raise an exception
                    assert isinstance(action, TeleportAction)
                env.step(action)


def test_different_observation_spaces():
    cfg = get_test_config(TEST_CFG_PATH)
    cfg.defrost()
    # these tasks have different obs spaces
    cfg.TASKS[0].SENSORS = ["PROXIMITY_SENSOR"]
    cfg.TASKS[1].SENSORS = [
        "POINTGOAL_WITH_GPS_COMPASS_SENSOR",
        "COMPASS_SENSOR",
        "PROXIMITY_SENSOR",
    ]
    # change after 1 episode
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_EPISODES = 1
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_CUM_STEPS = None
    cfg.CHANGE_TASK_BEHAVIOUR.LOOP = "FIXED"
    cfg.freeze()
    with MultiTaskEnv(config=cfg) as env:
        for _ in range(10):
            env.reset()
            # should always be in obs space
            assert "task_idx" in env.observation_space.spaces
            if env._curr_task_idx == 0:
                assert (
                    "POINTGOAL_WITH_GPS_COMPASS".lower()
                    not in env.observation_space.spaces
                )
                assert "PROXIMITY".lower() in env.observation_space.spaces
            elif env._curr_task_idx == 1:
                for sens in cfg.TASKS[1].SENSORS:
                    assert sens[:-7].lower() in env.observation_space.spaces

            while not env.episode_over:
                # execute random action
                env.step(env.action_space.sample())


def test_different_tasks():
    cfg = get_test_config(TEST_CFG_PATH)
    cfg.defrost()
    # add object nav task (requires additional dataset)
    cfg.TASKS[0].TYPE = ("ObjectNav-v1",)[0]
    cfg.TASKS[0].POSSIBLE_ACTIONS = (
        [
            "STOP",
            "MOVE_FORWARD",
            "TURN_LEFT",
            "TURN_RIGHT",
            "LOOK_UP",
            "LOOK_DOWN",
        ],
    )[0]
    cfg.TASKS[0].SUCCESS_DISTANCE = (0.1,)[0]
    cfg.TASKS[0].SENSORS = (
        ["OBJECTGOAL_SENSOR", "COMPASS_SENSOR", "GPS_SENSOR"],
    )[0]
    cfg.TASKS[0].GOAL_SENSOR_UUID = ("objectgoal",)[0]
    cfg.TASKS[0].MEASUREMENTS = (["DISTANCE_TO_GOAL", "SUCCESS", "SPL"],)
    cfg.TASKS[0].SUCCESS.SUCCESS_DISTANCE = (0.2,)[0]
    cfg.TASKS[0].DATASET.TYPE = ("ObjectNav-v1",)[0]
    cfg.TASKS[0].DATASET.SPLIT = ("val",)[0]
    cfg.TASKS[0].DATASET.CONTENT_SCENES = (["*"],)[0]
    cfg.TASKS[0].DATASET.DATA_PATH = (
        "data/datasets/objectnav/mp3d/v1/{split}/{split}.json.gz",
    )[0]
    cfg.TASKS[0].DATASET.SCENES_DIR = "data/scene_datasets/"

    if not os.path.exists(
        "data/datasets/objectnav/mp3d/v1/val/val.json.gz"
    ) or not os.path.exists("data/scene_datasets/mp3d/v1"):
        pytest.skip(
            "Please download Matterport3D scenes and ObjectNav Dataset to data folder to run this test."
        )

    # change after 1 episode
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_EPISODES = 1
    cfg.CHANGE_TASK_BEHAVIOUR.AFTER_N_CUM_STEPS = None
    cfg.CHANGE_TASK_BEHAVIOUR.LOOP = "FIXED"
    cfg.freeze()
    with MultiTaskEnv(config=cfg) as env:
        for _ in range(3):
            env.reset()
            # should always be in obs space
            assert "task_idx" in env.observation_space.spaces
            if env._curr_task_idx == 0:
                assert env.task.__class__.__name__ == "ObjectNavigationTask"
            elif env._curr_task_idx == 1:
                assert env.task.__class__.__name__ == "NavigationTask"

            while not env.episode_over:
                # execute random action
                env.step(env.action_space.sample())

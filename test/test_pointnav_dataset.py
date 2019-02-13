import time

import habitat
from habitat.config.default import cfg
from habitat.core.embodied_task import Episode
from habitat.core.logging import logger
from habitat.datasets import make_dataset
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

CFG_TEST = "test/pointnav_gibson_test.yaml"
PARTIAL_LOAD_SCENES = 3


def check_json_serializaiton(dataset: habitat.Dataset):
    start_time = time.time()
    json_str = str(dataset.to_json())
    logger.info(
        "JSON conversion finished. {} sec".format((time.time() - start_time))
    )
    decoded_dataset = dataset.__class__()
    decoded_dataset.from_json(json_str)
    assert len(decoded_dataset.episodes) > 0
    episode = decoded_dataset.episodes[0]
    assert isinstance(episode, Episode)
    assert (
        decoded_dataset.to_json() == json_str
    ), "JSON dataset encoding/decoding isn't consistent"


def test_single_pointnav_dataset():
    dataset_config = cfg(CFG_TEST).DATASET
    if not PointNavDatasetV1.check_config_paths_exist(dataset_config):
        logger.info("Test skipped as dataset files are missing.")
        return
    scenes = PointNavDatasetV1.get_scenes_to_load(config=dataset_config)
    assert len(scenes) == 0, (
        "Expected dataset doesn't expect separate " "episode file per scene."
    )
    dataset = PointNavDatasetV1(config=dataset_config)
    assert len(dataset.episodes) > 0, "The dataset shouldn't be empty."
    check_json_serializaiton(dataset)


def test_multiple_files_pointnav_dataset():
    dataset_config = cfg(CFG_TEST).DATASET
    dataset_config.SPLIT = "train"
    if not PointNavDatasetV1.check_config_paths_exist(dataset_config):
        logger.info("Test skipped as dataset files are missing.")
        return
    scenes = PointNavDatasetV1.get_scenes_to_load(config=dataset_config)
    assert len(scenes) > 0, (
        "Expected dataset contains separate episode  " "file per scene."
    )

    dataset_config.POINTNAVV1.CONTENT_SCENES = scenes[:PARTIAL_LOAD_SCENES]
    partial_dataset = make_dataset(
        id_dataset=dataset_config.TYPE, config=dataset_config
    )
    assert (
        len(partial_dataset.scene_ids) == PARTIAL_LOAD_SCENES
    ), "Number of loaded scenes doesn't correspond."
    check_json_serializaiton(partial_dataset)

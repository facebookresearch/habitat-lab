from habitat.config import get_crl_config
from habitat.config.default import _C
import pytest
import yaml

TEST_FILENAME = 'configs/test/habitat_cl_example.yaml'


def open_yaml(filename: str):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(filename: str, obj):
    with open('/tmp/{}'.format(filename), 'w') as f:
        yaml.dump(obj, f)


def test_first_task_gets_copied():
    cfg = get_crl_config(TEST_FILENAME)
    # 'TASKS' has a bit more stuff packed
    for k, _ in cfg.TASK.items():
        assert k in cfg.TASKS[0]


def test_tasks_keep_defaults():
    defaults = _C.TASK.clone()
    cfg = get_crl_config(TEST_FILENAME)
    cfg.defrost()
    cfg.TASKS[0].TYPE = 'MyCustomTestTask'
    cfg.freeze()
    assert cfg.TASKS[
        0].TYPE != cfg.TASK.TYPE, "Each tasks property should be overridable"
    for k in defaults.keys():
        for task in cfg.TASKS:
            assert k in task, "Default property should be inherithed by each task"


def test_global_dataset_config():
    datatype = 'MyDatasetType'
    config = open_yaml(TEST_FILENAME)
    for task in config['TASKS']:
        if 'DATASET' in task:
            del task['DATASET']

    config['DATASET']['TYPE'] = datatype
    save_yaml('test.yaml', config)
    # load test config
    cfg = get_crl_config('/tmp/test.yaml')
    # make sure each tasks has global dataset config
    for task in cfg.TASKS:
        assert task.DATASET.TYPE == cfg.DATASET.TYPE == datatype, 'Each task should inherit global dataset when dataset is not specified'


def test_global_dataset_config_override():
    datatype = 'MyDatasetType'
    datapath = '/some/path/'
    config = open_yaml(TEST_FILENAME)
    assert 'TASKS' in config
    assert len(config['TASKS']) > 0, 'Need at least one task in tasks to run test'
    for task in config['TASKS']:
        if 'DATASET' in task:
            del task['DATASET']
    # one tasks needs a different dataset
    config['TASKS'][0]['DATASET'] = {'TYPE': datatype, 'DATA_PATH': datapath}
    save_yaml('test.yaml', config)
    # load test config
    cfg = get_crl_config('/tmp/test.yaml')
    # make sure each tasks has global dataset config but the first one
    for i, task in enumerate(cfg.TASKS):
        if i == 0:
            assert task.DATASET.TYPE == datatype != cfg.DATASET.TYPE, 'First task should have a different dataset'
        else:
            assert task.DATASET.TYPE == cfg.DATASET.TYPE, 'Each task should inherit global dataset when dataset is not specified'
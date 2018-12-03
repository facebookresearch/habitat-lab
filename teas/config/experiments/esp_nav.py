from yacs.config import CfgNode


# TODO(akadian): Consider having a hierarchical config if the task definition
# gets more complex, similar to minos-eqa.
def esp_nav_cfg():
    config = CfgNode()
    config.resolution = (640, 480)
    config.hfov = '90'  # horizontal field of view in degrees
    config.seed = 100
    config.scene = 'data/esp/test.glb'
    config.max_episode_steps = 1000
    config.sensors = ['EspRGBSensor']
    config.sensor_position = [0, 0.05, 0]
    config.forward_step_size = 0.25  # in metres
    config.turn_angle = 10  # in degrees
    config.simulator = 'EspSimulator-v0'
    return config

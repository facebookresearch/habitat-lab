from yacs.config import CfgNode

# TODO(akadian): Consider having a hierarchical config if the task definition
# gets more complex, similar to minos-eqa.
def get_default_config():
    _esp_nav_c = CfgNode()
    _esp_nav_c.resolution = (640, 480)
    _esp_nav_c.hfov = '90'  # horizontal field of view in degrees
    _esp_nav_c.seed = 100
    _esp_nav_c.scene = 'data/esp/test.glb'
    _esp_nav_c.max_episode_steps = 1000
    _esp_nav_c.sensors = ['EspRGBSensor']
    _esp_nav_c.sensor_position = [0, 0.05, 0]
    _esp_nav_c.forward_step_size = 0.25  # in metres
    _esp_nav_c.turn_angle = 10  # in degrees
    _esp_nav_c.simulator = 'EspSimulator-v0'
    return _esp_nav_c

esp_nav_cfg = get_default_config()

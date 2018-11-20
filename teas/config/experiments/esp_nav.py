from yacs.config import CfgNode

# TODO(akadian): Consider having a hierarchical config if the task definition
# gets more complex, similar to minos-eqa.

_esp_nav_c = CfgNode()
_esp_nav_c.resolution = (640, 480)
_esp_nav_c.hfov = '90'  # horizontal field of view in degrees
_esp_nav_c.seed = 100
_esp_nav_c.scene = '/private/home/akadian/esp/test.glb'
_esp_nav_c.max_episode_steps = 1000
_esp_nav_c.sensors = ['EspRGBSensor']
_esp_nav_c.sensor_position = [0, 0.05, 0]
_esp_nav_c.forward_step_size = 0.25  # in metres
_esp_nav_c.turn_angle = 10  # in degrees
_esp_nav_c.simulator = 'EspSimulator-v0'

esp_nav_cfg = _esp_nav_c

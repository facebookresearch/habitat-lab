from yacs.config import CfgNode


# TODO(akadian): Consider having a hierarchical config if the task definition
# gets more complex, similar to minos-eqa.

def esp_nav_cfg():
    config = CfgNode()
    # Environment
    config.max_episode_steps = 1000

    # Simulator
    config.resolution = (640, 480)
    config.hfov = '90'  # horizontal field of view in degrees
    config.seed = 100
    config.scene = 'data/esp/test/test.glb'
    config.sensors = ['EspRGBSensor']
    config.sensor_position = [0, 0.05, 0]
    config.forward_step_size = 0.25  # in metres
    config.turn_angle = 10  # in degrees
    config.simulator = 'EspSimulator-v0'
    config.default_agent_id = 0

    # Agent configuration
    agent_c = CfgNode()
    agent_c.height = 1.5
    agent_c.radius = 0.1
    agent_c.mass = 32.0
    agent_c.linear_acceleration = 20.0
    agent_c.angular_acceleration = 4 * 3.14
    agent_c.linear_friction = 0.5
    agent_c.angular_friction = 1.0
    agent_c.coefficient_of_restitution = 0.0
    config.agents = [agent_c]

    return config

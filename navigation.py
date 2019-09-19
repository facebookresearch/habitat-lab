import numpy as np

import habitat
from habitat.sims import make_sim


class NavEnv:
    def __init__(self, forward_step, angle_step, base_controller):
        config = habitat.get_config()

        config.defrost()
        config.PYROBOT.BASE_CONTROLLER = base_controller
        config.freeze()

        self._reality = make_sim(id_sim="PyRobot-v0", config=config.PYROBOT)

        self._angle = (angle_step / 180) * np.pi

        self._actions = {
            "forward": [forward_step, 0, 0],
            "left": [0, 0, self._angle],
            "right": [0, 0, -self._angle],
            "stop": [0, 0, 0],
        }

    def reset(self):
        return self._reality.reset()

    @property
    def reality(self):
        return self._reality

    def step(self, action):
        if action not in self._actions:
            raise ValueError("Invalid action type {}".format(action))
        if action == "stop":
            raise NotImplementedError("stop action not implemented")

        observations = self._reality.step(
            "go_to_relative",
            {
                "xyt_position": self._actions[action],
                "use_map": False,
                "close_loop": True,
                "smooth": False
            }
        )

        return observations


def main():
    env = NavEnv()


if __name__ == "__main__":
    main()

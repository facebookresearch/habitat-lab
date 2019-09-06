from habitat.sims import make_sim
from PIL import Image


def main():
    reality = make_sim(
        id_sim="PyRobot-v0", config=None
    )
    print("Reality created")

    observations = reality.reset()
    observation = reality.step(0)

    print("RGB:", observations["rgb"].shape)
    print("Depth:", observations["depth"].shape)


if __name__ == "__main__":
    main()

import habitat
from habitat.sims import make_sim
import numpy as np
from PIL import Image


def main():
    config = habitat.get_config()
    config.freeze()

    reality = make_sim(
        id_sim="PyRobot-v0", config=config.PYROBOT
    )
    print("Reality created")

    observations = reality.reset()
    observations = reality.step(
        "go_to_relative",
        {
            "xyt_position": [0, 0, (10 / 180) * np.pi],
            "use_map": False,
            "close_loop": True,
            "smooth": False,
        }
    )

    depth = observations["depth"]
    import pdb; pdb.set_trace()
    depth_img = Image.fromarray((depth * 255).astype(np.uint8)[:, :, 0], mode="L")
    depth_img.save("/home/abhishek/Desktop/depth_img.png")

    rgb = observations["rgb"]
    rgb_img = Image.fromarray(rgb)
    rgb_img.save("/home/abhishek/Desktop/rgb_img.png")

    print("Depth:", observations["depth"].shape)
    print("RGB:", observations["rgb"].shape)


if __name__ == "__main__":
    main()

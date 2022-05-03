import os

# Video rendering utility.
from habitat_sim.utils import viz_utils as vut

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] ="quiet"
os.environ["HABITAT_SIM_LOG"] ="quiet"


# If the import block below fails due to an error like"'PIL.TiffTags' has no attribute
# 'IFD'", then restart the Colab runtime instance and rerun this cell and the previous cell.

# The ONLY two lines you need to add to start importing Habitat 2.0 Gym environments.
import gym

# flake8: noqa
import habitat.utils.gym_definitions


for name in [
        "HabitatPick-v0",
        "HabitatPlace-v0",
        "HabitatCloseCab-v0",
        "HabitatCloseFridge-v0",
        "HabitatOpenCab-v0",
        "HabitatOpenFridge-v0",
        "HabitatNavToObj-v0",
        "HabitatReachState-v0",
        "HabitatTidyHouse-v0",
        "HabitatPrepareGroceries-v0",
        "HabitatSetTable-v0",
        "HabitatNavPick-v0",
        "HabitatNavPickNavPlace-v0",
    ]:
    env = gym.make(name)
    video_file_path ="data/example_interact.mp4"
    video_writer = vut.get_fast_video_writer(video_file_path, fps=30)
    done = False
    env.reset()
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        video_writer.append_data(env.render("rgb_array"))
    video_writer.close()
    if vut.is_notebook():
        vut.display_video(video_file_path)
    env.close()
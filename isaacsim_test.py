#launch Isaac Sim before any other imports
#default first two lines in any standalone application

import time
import numpy as np

from habitat.isaac_sim.isaac_app_wrapper import IsaacAppWrapper


def convert_test():

    from omni.isaac.urdf import URDFImporter        

    pass

if __name__ == "__main__":

    headless = True
    import habitat
    # multiprocess_test()
    isaac_wrapper = IsaacAppWrapper(None, headless=True)

    convert_test()

    world = isaac_wrapper.service.world
    sim_app = isaac_wrapper.service.simulation_app
    spot_wrappers = []

    # spot_wrappers.append(SpotWrapper(world, "env_0", origin=np.array([0.0, 0.0, 0.0])))
    # src_spot_prim_path = spot_wrappers[0].get_prim_path()

    # env_origins = []
    # spacing = 15.0
    # num_rows = 1
    # num_cols = 1
    # for row in range(num_rows):
    #     for col in range(num_cols):
    #         env_origins.append(np.array([row * spacing, col * spacing, 0.0]))

    # for (i, origin) in enumerate(env_origins):
    #     env_id = f"env_{i + 1}"
    #     # temp disable instancing since it's broken
    #     spot_wrappers.append(SpotWrapper(world, env_id, src_prim_path=None, origin=origin))
    #     wrapper.add_kitchen_set(env_id, origin)

    # Resetting the world needs to be called before querying anything related to an articulation specifically.
    # Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
    world.reset()

    # world.step(render=False)
    if not headless:
        world.pause()

    # for spot_wrapper in spot_wrappers:
    #     spot_wrapper.post_reset()

    while sim_app.is_running():

        isaac_wrapper.step()

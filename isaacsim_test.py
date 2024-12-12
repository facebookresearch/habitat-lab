#launch Isaac Sim before any other imports
#default first two lines in any standalone application

import time
import numpy as np

from habitat.isaacsim.isaacsim_wrapper import IsaacSimWrapper, SpotWrapper

if __name__ == "__main__":

    headless = True
    import habitat
    # multiprocess_test()
    wrapper = IsaacSimWrapper(headless=headless, worker_name="0")

    world = wrapper.get_world()
    sim_app = wrapper.get_simulation_app()
    spot_wrappers = []

    # spot_wrappers.append(SpotWrapper(world, "env_0", origin=np.array([0.0, 0.0, 0.0])))
    # src_spot_prim_path = spot_wrappers[0].get_prim_path()

    env_origins = []
    spacing = 15.0
    num_rows = 1
    num_cols = 1
    for row in range(num_rows):
        for col in range(num_cols):
            env_origins.append(np.array([row * spacing, col * spacing, 0.0]))

    for (i, origin) in enumerate(env_origins):
        env_id = f"env_{i + 1}"
        # temp disable instancing since it's broken
        spot_wrappers.append(SpotWrapper(world, env_id, src_prim_path=None, origin=origin))
        wrapper.add_kitchen_set(env_id, origin)

    # Resetting the world needs to be called before querying anything related to an articulation specifically.
    # Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
    world.reset()
    # world.step(render=False)
    if not headless:
        world.pause()

    for spot_wrapper in spot_wrappers:
        spot_wrapper.post_reset()

    while sim_app.is_running():
        # Update the simulation app
        sim_app.update()

        # Add a short sleep to avoid maxing out CPU usage
        # time.sleep(0.01)

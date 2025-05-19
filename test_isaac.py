# flake8: noqa
# Must call this before importing Habitat or Magnum.
# fmt: off
import ctypes
import sys

sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

# fmt: on
import signal

import habitat_sim
from habitat.isaac_sim.isaac_app_wrapper import IsaacAppWrapper

try:
    from isaacsim import SimulationApp  # type: ignore
except Exception as e:
    print(f"Could not import Isaac bindings. Exception: '{e}'.")
    sys.exit(1)


def signal_handler(signum, frame):
    print(f"Signal {signum} received. Isaac crashed.")
    sys.exit(1)


signal.signal(signal.SIGSEGV, signal_handler)
signal.signal(signal.SIGABRT, signal_handler)

cfg_settings = habitat_sim.utils.settings.default_sim_settings.copy()
hab_cfg = habitat_sim.utils.settings.make_cfg(cfg_settings)
sim = habitat_sim.Simulator(hab_cfg)

try:
    isaac_wrapper = IsaacAppWrapper(sim, headless=True)
    for _ in range(10):
        print("Stepping.")
        isaac_wrapper.pre_render()
        isaac_wrapper.step()
except:
    print("Isaac could not be loaded.")
    sys.exit(1)

print("Isaac loaded.")
sys.exit(0)

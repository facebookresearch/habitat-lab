# mrp -f launch_hw.py

import mrp

mrp.import_msetup("agent")
mrp.import_msetup("hw")
mrp.import_msetup("sim")

# Metaprocs
mrp.process(
    name="hw_backend",
    deps=["roscore_hw", "stretch_core", "stretch_hector_slam"],
)

mrp.process(
    name="sim_backend",
    deps=["roscore_sim", "fake_stretch"],
)

mrp.process(
    name="hw_stack",
    deps=["hw_backend", "goto_controller", "state_estimator"],
)

mrp.process(
    name="sim_stack",
    deps=["sim_backend", "goto_controller", "state_estimator"],
)

if __name__ == "__main__":
    mrp.main()

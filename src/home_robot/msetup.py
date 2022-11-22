# mrp -f launch_hw.py

import mrp

mrp.import_msetup("agent")
mrp.import_msetup("hw")
mrp.import_msetup("sim")

# Metaprocs
mrp.process(name="agent_procs", deps=["goto_controller", "state_estimator"])

mrp.process(
    name="hw_stack",
    deps=["roscore_hw", "stretch_core", "stretch_hector_slam", "agent_procs"],
)

mrp.process(name="sim_stack", deps=["roscore_sim", "fake_stretch", "agent_procs"])

if __name__ == "__main__":
    mrp.main()

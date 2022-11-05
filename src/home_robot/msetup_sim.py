# mrp -f launch_hw.py

import mrp

mrp.import_msetup("agent")
mrp.import_msetup("sim/msetup_simple_sim.py")

if __name__ == "__main__":
    mrp.main()

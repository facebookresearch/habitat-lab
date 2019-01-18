from habitat.sims.habitat_sim import HabitatSimRGBSensor, \
    HabitatSimDepthSensor, HabitatSimSemanticSensor, HabitatSim
from habitat.sims.registration import sim_registry, \
    register_sim, \
    make_sim

register_sim(
    id_sim='Sim-v0',
    entry_point='habitat.sims.habitat_sim:HabitatSim')

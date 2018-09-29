from teas.simulators.minos import MinosSimulator
from teas.simulators.registration import simulator_registry, register_simulator, make_simulator

register_simulator(
    id_simulator='MinosSimulator-v0',
    entry_point=MinosSimulator)

from teas.simulators.registration import simulator_registry, \
    register_simulator, \
    make_simulator

register_simulator(
    id_simulator='MinosSimulator-v0',
    entry_point='teas.simulators.minos:MinosSimulator')

register_simulator(
    id_simulator='EspSimulator-v0',
    entry_point='teas.simulators.esp:EspSimulator')

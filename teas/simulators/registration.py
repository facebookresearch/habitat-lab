from teas.core.registry import Registry, Spec


class SimulatorSpec(Spec):
    def __init__(self, id_simulator, entry_point):
        super().__init__(id_simulator, entry_point)
        # TODO(akadian): Add more simulator specific details which will be
        # recorded to ensure reproducibility


class SimulatorRegistry(Registry):
    def register(self, id_simulator, **kwargs):
        if id_simulator in self.specs:
            raise ValueError(
                "Cannot re-register simulator specification with id: {}".format(
                    id_simulator))
        self.specs[id_simulator] = SimulatorSpec(id_simulator, **kwargs)


simulator_registry = SimulatorRegistry()


def register_simulator(id_simulator, **kwargs):
    simulator_registry.register(id_simulator, **kwargs)


def make_simulator(id_simulator, **kwargs):
    return simulator_registry.make(id_simulator, **kwargs)


def get_spec_simulator(id_simulator):
    return simulator_registry.get_spec(id_simulator)

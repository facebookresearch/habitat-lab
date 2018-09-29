from teas.core.registry import Registry, Spec


class DatasetSpec(Spec):
    def __init__(self, id_dataset, entry_point):
        super().__init__(id_dataset, entry_point)
        # TODO(akadian): Add more dataset specific details which will be
        # recorded to ensure reproducibility


class DatasetRegistry(Registry):
    def register(self, id_dataset, **kwargs):
        if id_dataset in self.specs:
            raise ValueError(
                "Cannot re-register dataset  specification with id: {}".format(
                    id_dataset))
        self.specs[id_dataset] = DatasetSpec(id_dataset, **kwargs)


dataset_registry = DatasetRegistry()


def register_dataset(id_dataset, **kwargs):
    dataset_registry.register(id_dataset, **kwargs)


def make_dataset(id_dataset, **kwargs):
    return dataset_registry.make(id_dataset, **kwargs)


def get_spec_dataset(id_dataset):
    return dataset_registry.get_spec(id_dataset)

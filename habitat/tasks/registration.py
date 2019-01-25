from habitat.core.logging import logger
from habitat.core.registry import Registry, Spec


class TaskSpec(Spec):
    def __init__(self, id_task, entry_point):
        super().__init__(id_task, entry_point)
        # TODO(akadian): Add more task specific details which will be
        # recorded to ensure reproducibility.


class TaskRegistry(Registry):
    def register(self, id_task, **kwargs):
        if id_task in self.specs:
            raise ValueError(
                "Cannot re-register task specification with id: {}".format(
                    id_task
                )
            )
        self.specs[id_task] = TaskSpec(id_task, **kwargs)


task_registry = TaskRegistry()


def register_task(id_task, **kwargs):
    task_registry.register(id_task, **kwargs)


def make_task(id_task, **kwargs):
    logger.info("initializing task {}".format(id_task))
    return task_registry.make(id_task, **kwargs)


def get_spec_task(id_task):
    return task_registry.get_spec(id_task)

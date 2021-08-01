import enum


class ActorWorkerTasks(enum.Enum):
    start = enum.auto()
    step = enum.auto()
    reset = enum.auto()
    set_transfer_buffers = enum.auto()


class PolicyWorkerTasks(enum.Enum):
    start = enum.auto()
    step = enum.auto()


class ReportWorkerTasks(enum.Enum):
    episode_end = enum.auto()
    learner_update = enum.auto()
    learner_timing = enum.auto()
    actor_timing = enum.auto()
    policy_timing = enum.auto()

from teas.tasks.eqa import MinosEqaTask
from teas.tasks.registration import task_registry, register_task, make_task

register_task(
    id_task='MinosEqa-v0',
    entry_point=MinosEqaTask)

from teas.tasks.registration import task_registry, register_task, make_task

register_task(
    id_task='MinosEQA-v0',
    entry_point='teas.tasks.eqa:MinosEQATask')

register_task(
    id_task='EQA-v0',
    entry_point='teas.tasks.eqa:EQATask')

register_task(
    id_task='Nav-v0',
    entry_point='teas.tasks.nav:NavigationTask')

from teas.tasks.registration import task_registry, register_task, make_task

register_task(
    id_task='MinosEqa-v0',
    entry_point='teas.tasks.eqa:MinosEqaTask')

register_task(
    id_task='EspNav-v0',
    entry_point='teas.tasks.nav:EspNavToy')

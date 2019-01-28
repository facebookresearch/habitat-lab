from habitat.tasks.registration import task_registry, register_task, make_task

register_task(id_task="EQA-v0", entry_point="habitat.tasks.eqa:EQATask")

register_task(id_task="Nav-v0", entry_point="habitat.tasks.nav:NavigationTask")

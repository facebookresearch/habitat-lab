
from habitat.sims.habitat_simulator.actions import HabitatSimActions

def _try_register_ovmm_task():
    import habitat.tasks.ovmm.ovmm_sensors
    import habitat.tasks.ovmm.sub_tasks.nav_to_obj_sensors
    import habitat.tasks.ovmm.sub_tasks.nav_to_obj_task
    import habitat.tasks.ovmm.sub_tasks.place_sensors

    if not HabitatSimActions.has_action("manipulation_mode"):
        HabitatSimActions.extend_action_space("manipulation_mode")

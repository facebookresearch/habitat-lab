from habitat.tasks.nav.nav_task import StopAction


def sample_non_stop_action(action_space, num_samples=1):
    samples = []
    for i in range(num_samples):
        action_opts = action_space.sample()
        while action_opts["action"] == StopAction.name:
            action_opts = action_space.sample()
        samples.append(action_opts)
    print(samples)
    if num_samples == 1:
        return samples[0]
    else:
        return samples
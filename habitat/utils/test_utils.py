from habitat.tasks.nav.nav_task import StopAction


def sample_non_stop_action(action_space, num_samples=1):
    samples = []
    for i in range(num_samples):
        action = action_space.sample()
        while action["action"] == StopAction.name:
            action = action_space.sample()
        samples.append(action)
    print(samples)
    if num_samples == 1:
        return samples[0]
    else:
        return samples

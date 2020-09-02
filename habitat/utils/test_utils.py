from habitat.tasks.nav.nav import StopAction


def sample_non_stop_action(action_space, num_samples=1):
    samples = []
    for _ in range(num_samples):
        action = action_space.sample()
        while action["action"] == StopAction.name:
            action = action_space.sample()
        samples.append({"action": action})

    if num_samples == 1:
        return samples[0]["action"]
    else:
        return samples

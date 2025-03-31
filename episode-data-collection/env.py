import habitat

class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        dist = observations["pointgoal_with_gps_compass"][0]

        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

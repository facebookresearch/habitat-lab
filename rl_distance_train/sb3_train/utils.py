from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class SyncEvalEnvStatsCallback(BaseCallback):
    def __init__(self, train_norm, eval_norm, sync_freq, verbose=0):
        super().__init__(verbose)
        self.train_norm = train_norm
        self.eval_norm = eval_norm
        self.sync_freq = sync_freq

    def _on_step(self) -> bool:
        # Sync the stats every `sync_freq` steps
        if self.num_timesteps % self.sync_freq == 0:
            self.eval_norm.obs_rms = self.train_norm.obs_rms
            self.eval_norm.ret_rms = self.train_norm.ret_rms
            self.eval_norm.clip_obs = self.train_norm.clip_obs
            self.eval_norm.clip_reward = self.train_norm.clip_reward
            self.eval_norm.training = False
            self.eval_norm.norm_reward = False
            if self.verbose > 0:
                print("Synchronized normalization statistics from training to evaluation environment.")
                
        return True
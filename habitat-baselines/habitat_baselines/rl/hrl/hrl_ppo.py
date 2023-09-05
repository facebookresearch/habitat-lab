import torch
import torch.nn.functional as F

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.algo.ddppo import DecentralizedDistributedMixin
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.utils.common import (
    LagrangeInequalityCoefficient,
    inference_mode,
)


@baseline_registry.register_updater
class HRLPPO(PPO):
    def _update_from_batch(self, batch, epoch, rollouts, learner_metrics):
        n_samples = max(batch["loss_mask"].sum(), 1)

        def record_min_mean_max(t: torch.Tensor, prefix: str):
            for name, op in (
                ("min", torch.min),
                ("mean", torch.mean),
                ("max", torch.max),
            ):
                learner_metrics[f"{prefix}_{name}"].append(op(t))

        def reduce_loss(loss):
            return (loss * batch["loss_mask"]).sum() / n_samples

        self._set_grads_to_none()

        (
            values,
            action_log_probs,
            dist_entropy,
            _,
            aux_loss_res,
        ) = self._evaluate_actions(
            batch["observations"],
            batch["recurrent_hidden_states"],
            batch["prev_actions"],
            batch["masks"],
            batch["actions"],
            batch["rnn_build_seq_info"],
        )

        ratio = torch.exp(action_log_probs - batch["action_log_probs"])

        surr1 = batch["advantages"] * ratio
        surr2 = batch["advantages"] * (
            torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
        )
        action_loss = -torch.min(surr1, surr2)
        action_loss = reduce_loss(action_loss)

        values = values.float()
        orig_values = values

        if self.use_clipped_value_loss:
            delta = values.detach() - batch["value_preds"]
            value_pred_clipped = batch["value_preds"] + delta.clamp(
                -self.clip_param, self.clip_param
            )

            values = torch.where(
                delta.abs() < self.clip_param,
                values,
                value_pred_clipped,
            )

        value_loss = 0.5 * F.mse_loss(
            values, batch["returns"], reduction="none"
        )
        value_loss = reduce_loss(value_loss)

        all_losses = [
            self.value_loss_coef * value_loss,
            action_loss,
        ]
        all_losses.extend(v["loss"] for v in aux_loss_res.values())

        dist_entropy = reduce_loss(dist_entropy)
        if isinstance(self.entropy_coef, float):
            all_losses.append(-self.entropy_coef * dist_entropy)
        else:
            all_losses.append(self.entropy_coef.lagrangian_loss(dist_entropy))

        total_loss = torch.stack(all_losses).sum()

        total_loss = self.before_backward(total_loss)
        total_loss.backward()
        self.after_backward(total_loss)

        grad_norm = self.before_step()
        self.optimizer.step()
        self.after_step()

        with inference_mode():
            record_min_mean_max(orig_values, "value_pred")
            record_min_mean_max(ratio, "prob_ratio")
            total_size = batch["loss_mask"].shape[0]
            if isinstance(n_samples, torch.Tensor):
                n_samples = n_samples.item()
            learner_metrics["batch_filled_ratio"].append(
                n_samples / total_size
            )

            learner_metrics["value_loss"].append(value_loss)
            learner_metrics["action_loss"].append(action_loss)
            learner_metrics["dist_entropy"].append(dist_entropy)
            if epoch == (self.ppo_epoch - 1):
                learner_metrics["ppo_fraction_clipped"].append(
                    (ratio > (1.0 + self.clip_param)).float().mean()
                    + (ratio < (1.0 - self.clip_param)).float().mean()
                )

            learner_metrics["grad_norm"].append(grad_norm)
            if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
                learner_metrics["entropy_coef"].append(
                    self.entropy_coef().detach()
                )
            for name, res in aux_loss_res.items():
                for k, v in res.items():
                    learner_metrics[f"aux_{name}_{k}"].append(v.detach())


@baseline_registry.register_updater
class HRLDDPPO(DecentralizedDistributedMixin, HRLPPO):
    pass

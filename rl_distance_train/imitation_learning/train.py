# run_bc.py

import os
import time
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from habitat import logger as habitat_logger
from habitat import Env
from habitat_baselines.common.baseline_registry import baseline_registry

from rl_distance_train.imitation_learning.data_loader import DataBuffer
from habitat.tasks.nav.nav import ImageGoalSensor

DATA_PATH = "/cluster/scratch/lmilikic/hm3d/episodes/"
BATCH_SIZE = 8

# -----------------------------------------------------------------------------
def setup_logging(log_dir: str):
    """Configure Python logging (console + file) and return (logger, tb_writer)."""
    os.makedirs(log_dir, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-5s %(message)s"
    # root logger writes both to console and to file
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "train.log")),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    tb_writer = SummaryWriter(log_dir)
    return logging.getLogger(), tb_writer

# -----------------------------------------------------------------------------
def run_bc(cfg: DictConfig):
    # 1) Hydra + Python logger + TB
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_root  = getattr(cfg, "output_root", os.getcwd())
    log_dir   = os.path.join(out_root, cfg.habitat_baselines.tensorboard_dir, timestamp)
    logger, tb_writer = setup_logging(log_dir)

    # also echo out full Hydra config
    logger.info("Hydra config:\n" + OmegaConf.to_yaml(cfg))

    # 2) device
    device = torch.device(f"cuda:{cfg.habitat_baselines.torch_gpu_id}"
                          if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 3) build a dummy env to grab spaces
    with Env(config=cfg.habitat) as tmp_env:
        obs_space = tmp_env.observation_space
        act_space = tmp_env.action_space
        obs_space.spaces.pop("compass", None)
        obs_space.spaces.pop("gps",     None)

    # 4) policy + model
    policy_cls = baseline_registry.get_policy(
        cfg.habitat_baselines.rl.policy.main_agent.name
    )
    policy = policy_cls.from_config(cfg, observation_space=obs_space, action_space=act_space)
    policy = policy.to(device)
    model  = policy.net.to(device)

    # 5) optimizer, scheduler, criterion
    total_steps = int(cfg.habitat_baselines.total_num_steps)
    log_interval = int(cfg.habitat_baselines.log_interval)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=cfg.habitat_baselines.rl.ppo.lr,
        weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, total_steps // log_interval),
        gamma=0.5,
    )

    # 6) data loaders
    train_ds = DataBuffer(
        datapath=os.path.join(DATA_PATH, "train_episodes"),
        doaug=True, normalize_img=True
    )
    val_ds = DataBuffer(
        datapath=os.path.join(DATA_PATH, "val_episodes"),
        doaug=False, normalize_img=True
    )
    train_loader = iter(DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True))
    val_loader   = iter(DataLoader(val_ds, batch_size=2*BATCH_SIZE, num_workers=4, pin_memory=True))

    # 7) RNN flags
    is_lstm     = (cfg.habitat_baselines.rl.ddppo.rnn_type == "LSTM")
    num_layers  = model.state_encoder.rnn.num_layers
    hidden_size = model.state_encoder.rnn.hidden_size

    # scheduled sampling
    decay_steps = total_steps

    best_acc = 0.0

    # 8) main training loop

    ## prepare save checkpoint
    ckpt_dir = cfg.habitat_baselines.eval_ckpt_path_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "config.yaml")
    OmegaConf.save(cfg, ckpt_path)

    pbar = tqdm(range(1, total_steps+1), desc="BC steps")
    for step in pbar:
        model.train()
        traj_ims, goal_img, actions = next(train_loader)
        B, T = actions.shape

        traj_ims, goal_img, actions = [x.to(device) for x in (traj_ims, goal_img, actions)]

        # init hidden & prev/masks
        if is_lstm:
            h0 = torch.zeros(B, num_layers, hidden_size, device=device)
            c0 = torch.zeros(B, num_layers, hidden_size, device=device)
            hidden_states = torch.cat((h0, c0), dim=1)
        else:
            hidden_states = torch.zeros(B, num_layers, hidden_size, device=device)

        prev_actions = -torch.ones(B, device=device, dtype=torch.int64)
        masks        = torch.ones(B, device=device, dtype=torch.bool)

        optimizer.zero_grad()
        total_loss = 0.0

        # unroll
        for t in tqdm(range(T), leave=False, desc="Trajectory steps"):
            obs = {
                "rgb": traj_ims[:, t],
                ImageGoalSensor.cls_uuid: goal_img,
            }
            feats, hidden_states, _ = model(obs, hidden_states, prev_actions, masks)
            dist   = policy.action_distribution(feats)
            logits = dist.logits

            total_loss += criterion(logits, actions[:, t])

            next_pa = actions[:, t]
            prev_actions = next_pa

            done  = (actions[:, t] == 0)
            masks = ~done

        loss = total_loss / T
        loss.backward()
        optimizer.step()
        scheduler.step()

        # TensorBoard scalars
        tb_writer.add_scalar("train/loss", loss.item(), step)
        tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

        # validation + logging
        if step % log_interval == 0:
            model.eval()
            with torch.no_grad():
                v_traj, v_goal, v_actions = next(val_loader)
                Bv, Tv = v_actions.shape
                v_traj, v_goal, v_actions = [x.to(device) for x in (v_traj, v_goal, v_actions)]

                if is_lstm:
                    vh0 = torch.zeros(Bv, num_layers, hidden_size, device=device)
                    vc0 = torch.zeros(Bv, num_layers, hidden_size, device=device)
                    v_hidden = torch.cat((vh0, vc0), dim=1)
                else:
                    v_hidden = torch.zeros(Bv, num_layers, hidden_size, device=device)

                v_prev_actions = -torch.ones(Bv, device=device, dtype=torch.int64)
                v_masks        = torch.ones(Bv, dtype=torch.bool, device=device)

                v_loss  = 0.0
                correct = 0
                for t in tqdm(range(Tv), leave=False, desc="Trajectory steps"):
                    obs = {
                        "rgb": v_traj[:, t],
                        ImageGoalSensor.cls_uuid: v_goal,
                    }
                    feats, v_hidden, _ = model(obs, v_hidden, v_prev_actions, v_masks)
                    dist       = policy.action_distribution(feats)
                    logits     = dist.logits
                    v_loss    += criterion(logits, v_actions[:, t])
                    preds      = logits.argmax(-1)
                    correct   += (preds == v_actions[:, t]).sum().item()

                    done = (v_actions[:, t] == 0)
                    v_prev_actions = v_actions[:, t]
                    v_masks = ~done

                v_loss = (v_loss / Tv).item()
                v_acc  = correct / (Bv * Tv)

            # Python logging
            logger.info(f"[{step}/{total_steps}] val_loss={v_loss:.4f} val_acc={100*v_acc:.2f}%")

            # TensorBoard
            tb_writer.add_scalar("val/loss", v_loss, step)
            tb_writer.add_scalar("val/acc", v_acc, step)

            # tqdm postfix
            pbar.set_postfix(
                train_loss=f"{loss:.3f}",
                val_loss=f"{v_loss:.3f}",
                val_acc=f"{100*v_acc:.1f}%"
            )

            # save best
            if v_acc > best_acc:
                best_acc = v_acc
                ckpt_path = os.path.join(ckpt_dir, "best_dense.pth")
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"→ New best model saved @ {ckpt_path} ({100*best_acc:.2f}%)")

    tb_writer.close()
    logger.info("Training complete.")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_bc()

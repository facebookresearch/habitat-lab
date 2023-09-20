import argparse
import os
import random
import string
import uuid


def zsc_eval_single(plan_idx=None, learned_agent=None, add_opts=None):
    rnd_id = random.choice(string.ascii_uppercase) + str(uuid.uuid4())[:8]

    if plan_idx is not None:
        run_cmd = f"python habitat-baselines/habitat_baselines/run.py -m --config-name=experiments_hab3/pop_play_kinematic_oracle_humanoid_spot_fp_learned_skill.yaml habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.plan_idx={plan_idx} {add_opts}"
        print(f"RUNNING {run_cmd}")
        os.system(run_cmd)

    if learned_agent is not None:
        run_cmd = f"python habitat-baselines/habitat_baselines/run.py -m --config-name=experiments_hab3/pop_play_kinematic_oracle_humanoid_spot_fp_learned_skill_learn.yaml habitat_baselines.rl.agent.load_type1_pop_ckpts=[{learned_agent}] {add_opts}"
        print(f"RUNNING {run_cmd}")
        os.system(run_cmd)


def zsc_eval(args, add_opts):
    rnd_id = random.choice(string.ascii_uppercase) + str(uuid.uuid4())[:8]

    for plan_idx in range(-args.num_plans, 0):
        zsc_eval_single(plan_idx, None, add_opts)
        if args.debug:
            break

    learned_agents = args.learned_agents.split(",")
    for learned_agent in learned_agents:
        zsc_eval_single(None, learned_agent, add_opts)
        if args.debug:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # For tidy-house style task there are 4 plans.
    parser.add_argument("--num-plans", type=int, default=4)
    parser.add_argument("--plan_idx", type=int, default=None)
    parser.add_argument(
        "--learned-agents",
        type=str,
        default=None,
        help="Comma separated list of checkpoints of holdout learned agents.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    add_opts = " ".join(args.opts)

    # zsc_eval(args, add_opts)
    zsc_eval_single(args.plan_idx, args.learned_agents, add_opts)

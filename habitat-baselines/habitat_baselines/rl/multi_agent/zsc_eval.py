import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # For tidy-house there are 5 plans.
    parser.add_argument("--num-plans", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    add_opts = " ".join(args.opts)

    for plan_idx in range(args.num_plans):
        run_cmd = f"python habitat_baselines/run.py --config-name=experiments_hab3/eval_zsc_kinematic_oracle.yaml habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.plan_idx={plan_idx} {add_opts}"
        print(f"RUNNING {run_cmd}")
        os.system(run_cmd)
        if args.debug:
            break

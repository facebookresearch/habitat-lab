import argparse
import os
import random
import string
import uuid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # For tidy-house style task there are 5 plans.
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cmd", type=str, default="bdp_or")
    parser.add_argument("--proj-dat", type=str, required=True)
    parser.add_argument("--runs", type=str, required=True)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    add_opts = " ".join(args.opts)

    rnd_id = random.choice(string.ascii_uppercase) + str(uuid.uuid4())[:8]

    for agent_idx in range(args.num_agents):
        run_cmd = f"python -m rl_utils.launcher.eval_sys --cfg ~/configs/hab3.yaml --cmd {args.cmd} --runs {args.runs} --proj-dat {args.proj_dat} habitat_baselines.rl.agent.force_partner_sample_idx={agent_idx} {add_opts} habitat_baselines.wb.group={rnd_id}"
        print(f"RUNNING {run_cmd}")
        os.system(run_cmd)
        if args.debug:
            break

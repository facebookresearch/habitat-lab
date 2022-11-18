import argparse
import pandas as pd
from omegaconf import OmegaConf
from rl_utils.plotting.wb_query import query
from rl_utils.plotting.auto_line import line_plot
from rl_utils.plotting.utils import fig_save
from functools import reduce

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--h2-name", type=str, required=True)
    argparse.add_argument("--cache", action="store_true")
    args = argparse.parse_args()

    cfg = OmegaConf.load("/coc/testnvme/aszot3/configs/habitat.yaml")

    renames = {
        "metrics/did_pick_object": "Did Pick Object Ratio",
        "reward": "Episode Return",
    }

    all_succ_k = [f"ALL_{x}" for x in renames.keys()]
    all_time_k = "ALL__runtime"
    time_k = "_runtime"

    all_result = query(
        [*all_succ_k, all_time_k],
        {"WB.RUN_NAME": args.h2_name},
        cfg,
        use_cached=args.cache,
        verbose=False,
    )
    all_result = [
        reduce(
            lambda x, y: pd.merge(x, y, on="_step"),
            [x for x in result.values() if isinstance(x, pd.DataFrame)],
        )
        for result in all_result
    ]

    all_result = [df.set_index("_step") for df in all_result]
    for i in reversed(range(len(all_result) - 1)):
        all_result[i][time_k] += all_result[i][time_k].iloc[-1]
    df = pd.concat(reversed(all_result))
    df = df.reset_index(level=0)
    df["method"] = "h2"
    # It is in seconds
    df[time_k] /= 60

    df["rank"] = 0

    for metric, name in renames.items():
        fig, ax = line_plot(
            df,
            "_step",
            metric,
            "rank",
            "method",
            smooth_factor={"h2": 0.0},
            # y_disp_bounds=[0, 100.0],
            # x_disp_bounds=[0, max_time],
            legend=False,
            rename_map={
                **renames,
                "_step": "Environment Steps",
                metric: name,
                time_k: "Time (Minutes)",
                "h2": "Habitat 2.0",
                "gala": "Galactic",
            },
            num_marker_points={
                "h2": 0,
                "gala": 0,
            },
            x_logscale=False,
            legend_loc="Upper left",
        )
        metric_name = metric.split("/")[-1]
        fig_save("data/vis", f"h2_{metric_name}", fig)

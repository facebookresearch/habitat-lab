import argparse
import pandas as pd
from omegaconf import OmegaConf
from rl_utils.plotting.wb_query import query
from rl_utils.plotting.auto_line import line_plot
from rl_utils.plotting.utils import fig_save

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--h2-name", type=str, required=True)
    argparse.add_argument("--cache", action="store_true")
    args = argparse.parse_args()

    cfg = OmegaConf.load("/coc/testnvme/aszot3/configs/habitat.yaml")

    all_succ_k = "ALL_metrics/rearrangepick_success"
    all_time_k = "ALL__runtime"
    succ_k = "metrics/rearrangepick_success"
    time_k = "_runtime"

    all_result = query(
        [all_succ_k, all_time_k],
        {"WB.RUN_NAME": args.h2_name},
        cfg,
        use_cached=args.cache,
        verbose=False,
    )

    all_result = [
        result[all_succ_k].join(result[all_time_k][time_k])
        for result in all_result
    ]
    all_result = [df.set_index("_step") for df in all_result]
    for i in reversed(range(len(all_result) - 1)):
        all_result[i][time_k] += all_result[i + 1][time_k].iloc[-1]
    df = pd.concat(reversed(all_result))
    df = df.reset_index(level=0)
    df["method"] = "h2"
    # It is in seconds
    df[time_k] /= 60
    max_time = df[time_k].iloc[-1]

    df_g = pd.read_csv("data/vis/mobile_pick_train_resnet18.csv")
    df_g = df_g.rename({"Time (minutes)": time_k, " success": succ_k}, axis=1)
    # It is in minutes
    # df_g[time_k] /= 60.0
    df_g["method"] = "gala"

    first_g = df_g[df_g[succ_k] >= 0.8].iloc[0][time_k]
    first_h = df[df[succ_k] >= 0.8].iloc[0][time_k]
    print(f"Gala {first_g}")
    print(f"H2 {first_h}")
    print(f"Factor {first_h / first_g}")

    df = pd.concat([df, df_g])
    df["rank"] = 0
    df[succ_k] *= 100.0
    fig, ax = line_plot(
        df,
        time_k,
        succ_k,
        "rank",
        "method",
        smooth_factor={"h2": 0.0},
        y_disp_bounds=[0, 100.0],
        # x_disp_bounds=[0, max_time],
        legend=True,
        rename_map={
            succ_k: "Success Rate (%)",
            time_k: "Time (Minutes)",
            "h2": "Habitat 2.0",
            "gala": "Galactic",
        },
        num_marker_points={
            "h2": 0,
            "gala": 0,
        },
        x_logscale=True,
        legend_loc="lower left",
    )

    for x in [first_g, first_h]:
        ax.axvline(
            x=x, color="gray", alpha=0.5, linestyle="solid", linewidth=1.5
        )
    fig_save("data/vis", "pick_time", fig)

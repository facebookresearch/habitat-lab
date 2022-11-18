import argparse
import pandas as pd
from rl_utils.plotting.auto_table import plot_table
from omegaconf import OmegaConf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    all_data = cfg.data
    df = pd.DataFrame(
        [
            {
                "method": method_name,
                "perf": row[0],
                "err": row[1],
                "rank": "0",
                "setting": row_name,
            }
            for method_name, data in all_data.items()
            for row_name, row in zip(["train", "val"], data)
        ]
    )

    plot_table(
        df,
        "method",
        "setting",
        "perf",
        col_order=list(all_data.keys()),
        row_order=["train", "val"],
        renames={
            "train": "Train",
            "val": "Eval",
        },
        err_key="err",
        write_to="/Users/andrewszot/Documents/writing/gala_cvpr2023/tables/sim2sim.tex",
    )

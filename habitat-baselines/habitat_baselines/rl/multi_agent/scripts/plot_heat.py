import argparse
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rl_utils.plotting.utils as putils

EVENT_NAME = "Event"
RENAME_MAP = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, type=str)
    args = parser.parse_args()


def plot_heatmap(
    arr,
    save_name,
    agent_names,
    event_order,
    save_dir,
    xlabel,
    cmap,
    method,
    title=None,
    xlabels=None,
):
    event_names = [RENAME_MAP.get(x, x) for x in event_order]
    cmap = matplotlib.cm.get_cmap(cmap)
    fig, ax = plt.subplots()
    neg = ax.imshow(arr, interpolation="none", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(agent_names)))
    if xlabels is not None:
        ax.set_xticklabels(xlabels)

    ax.set_yticks(np.arange(len(event_names)))
    ax.set_yticklabels(event_names)

    ax.set_ylabel(EVENT_NAME)
    ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)

    save_dir = f"data/vis/{save_dir}/"
    full_path = osp.join(save_dir, f"{save_name}_{method}.png")
    fig.savefig(full_path, bbox_inches="tight")

    fig.colorbar(neg, ax=ax, orientation="horizontal")
    ax.remove()
    putils.fig_save(
        save_dir,
        "cbar",
        fig,
        is_high_quality=True,
        verbose=False,
        clear=True,
        log_wandb=False,
    )


if __name__ == "__main__":
    main()

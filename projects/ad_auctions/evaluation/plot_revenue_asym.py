import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from itertools import combinations_with_replacement
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.ad_auctions.config_exp import *
from projects.ad_auctions.evaluation.util import *
from soda.game import Game
from soda.strategy import Strategy
from soda.util.evaluation import get_results

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [COLORS[0], "#ffffff"])


def get_name_setting(payment_rule, agent1, agent2, prior) -> str:
    assert prior in ["gaussian", "uniform"]
    return f"{payment_rule}2_{agent1}_{agent2}" + (
        "_gaus" if prior == "gaussian" else ""
    )


def plot_revenue_asym(path_to_exp, path_save, prior: str = "uniform"):
    """create plot (matrix) for difference in utility for asymmetric utility functions"""
    assert prior in ["uniform", "gaussian"]
    ## get data
    # import logfiles (remove duplicated since revenue is the same for agents)
    df = get_revenue(path_to_exp, "revenue_asym")
    df = df[["setting", "metric", "mean", "std"]].drop_duplicates()

    # compute rel. diff. in revenue
    util_type = ["ql", "roi", "ros"]
    labels = ["QL", "ROI", "ROSB"]
    agents = list(combinations_with_replacement(util_type, 2))
    matrix = np.nan * np.zeros((3, 3))
    for agent1, agent2 in agents:
        revenue_fp = df[df.setting == get_name_setting("fp", agent1, agent2, prior)][
            "mean"
        ].item()
        revenue_sp = df[df.setting == get_name_setting("sp", agent1, agent2, prior)][
            "mean"
        ].item()
        matrix[util_type.index(agent1), util_type.index(agent2)] = (
            revenue_fp / revenue_sp - 1
        ) * 100

    ## create plot ##
    # create axis
    fig, ax = set_axis(xlabel="", ylabel="")
    ax.grid(visible=False)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks(
        range(3),
        labels,
        rotation=0,
        fontsize=PARAM["fontsize_label"] - 1,
    )
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks(
        range(3),
        labels,
        rotation=0,
        fontsize=PARAM["fontsize_label"] - 1,
    )

    # create image plot with colorbar
    img = ax.imshow(matrix, origin="lower", vmin=-45, vmax=5, cmap=cmap)
    cbar = plt.colorbar(img, ax=ax, shrink=0.63)
    cbar.set_label("Rel. Difference in Revenue [%]", fontsize=PARAM["fontsize_legend"])
    cbar.set_ticks([-40, -30, -20, -10, 0])

    # add numbers
    for agent1, agent2 in agents:
        revenue_fp = df[df.setting == get_name_setting("fp", agent1, agent2, prior)][
            "mean"
        ].item()
        revenue_sp = df[df.setting == get_name_setting("sp", agent1, agent2, prior)][
            "mean"
        ].item()

        ax.text(
            util_type.index(agent2),
            util_type.index(agent1),
            s=f"FP: {revenue_fp:.2f}",
            ha="center",
            va="top",
            fontsize=PARAM["fontsize_legend"],
        )
        ax.text(
            util_type.index(agent2),
            util_type.index(agent1),
            s=f"SP: {revenue_sp:.2f}",
            ha="center",
            va="bottom",
            fontsize=PARAM["fontsize_legend"],
        )

    # save plot
    fig.savefig(f"{path_save}/revenue_asym_{prior}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    LABEL_EXPERIMENT = "revenue_asym"
    path_save = os.path.join(PATH_TO_RESULTS, LABEL_EXPERIMENT)
    os.makedirs(path_save, exist_ok=True)
    plot_revenue_asym(PATH_TO_EXPERIMENTS, path_save, prior="uniform")
    plot_revenue_asym(PATH_TO_EXPERIMENTS, path_save, prior="gaussian")

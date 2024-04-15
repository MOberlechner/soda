import os
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

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [COLORS[1], COLORS[2]])
parameter = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
COLORS_ROIS = [cmap(p) for p in parameter]


def plot_revenue(path_to_configs, path_to_experiments, path_save):

    labels = ["0 (ROI)", "1/4", "1/2", "3/4", "1 (ROS)"]
    learner = "soda1_revenue"
    n_bidder = 2

    df = get_revenue(path_to_experiments, "revenue_rois")
    revenue = {}
    for payment_rule in ["fp", "sp"]:
        settings = [f"rois{k+1}_{payment_rule}_{n_bidder}" for k in range(5)]
        revenue[payment_rule] = [
            df.loc[(df.setting == s) & (df.learner == learner), "mean"].item()
            for s in settings
        ]

    fig, ax = set_axis("Parameter $\mu$", "Revenue")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(0, 1.01)
    # plot revenue
    ax.bar(parameter - 0.055, revenue["fp"], width=0.1, color=COLORS_ROIS)
    ax.bar(parameter + 0.055, revenue["sp"], width=0.1, color=COLORS_ROIS, alpha=0.5)
    ax.bar(
        parameter + 0.055,
        revenue["sp"],
        width=0.1,
        facecolor="none",
        hatch="///",
        edgecolor=COLORS_ROIS,
    )
    ax.set_xticks(parameter, labels)
    # legend
    ax.bar([-1], [1], label="First-Price", facecolor="white", edgecolor="k")
    ax.bar([-1], [1], label="Second-Price", hatch="//", color="white", edgecolor="k")
    ax.legend(fontsize=PARAM["fontsize_legend"], loc=2)
    fig.savefig(f"{path_save}/revenue_rois.pdf", bbox_inches="tight")


if __name__ == "__main__":

    EXPERIMENT_TAG = "revenue_rois"
    path_save = f"{PATH_SAVE}revenue_rois/"
    os.makedirs(path_save, exist_ok=True)

    config_learner = "soda1_revenue.yaml"
    n_bidder = 2

    # plot revenue
    plot_revenue(PATH_TO_CONFIGS, PATH_TO_EXPERIMENTS, path_save)

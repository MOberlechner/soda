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


def plot_strategies_revenue(
    budget,
    payment_rule,
    path_to_configs,
    path_to_experiments,
    path_save,
    tag: str = "",
):

    config_games = [
        f"revenue_budget/{util}_{payment_rule}_{n_bidder}_b{budget}.yaml"
        for util in ["ql", "roi", "ros"]
    ]
    labels = ["QL", "ROI", "ROSB"]

    fig, ax = set_axis("Valuation v", "Bid b")
    for i in range(3):
        config_game = config_games[i]
        label = labels[i]
        color = COLORS[i]

        # get data from computed strategies
        game, _, strategies = get_results(
            config_game,
            config_learner,
            path_to_configs,
            path_to_experiments,
            experiment_tag="revenue_budget",
            run=0,
        )
        x, bids_mean, bids_std, bne = get_bids(game, strategies, agent="1")

        # plot results
        ax.plot(x, bids_mean, color=color, linestyle="-", label=label, linewidth=2)
        ax.fill_between(
            x, bids_mean - bids_std, bids_mean + bids_std, alpha=0.4, color=color
        )

    ax.legend(fontsize=PARAM["fontsize_legend"], loc=2)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 1)
    fig.savefig(
        f"{path_save}/revenue_budget{budget}_strat_{tag}{payment_rule}_{game.n_bidder}.pdf",
        bbox_inches="tight",
    )


def plot_revenue(budget: int, path_to_configs, path_to_experiments, path_save):

    labels = ["0.0\nROI", "1/4", "1/2", "3/4", "1.0\nROS"]
    learner = "soda1_revenue"
    n_bidder = 2

    df = get_revenue(path_to_experiments, "revenue_rois")
    revenue = {}
    for payment_rule in ["fp", "sp"]:
        settings = [f"rois{k+1}_{payment_rule}_2_b{budget}" for k in range(5)]
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
    fig.savefig(f"{path_save}/revenue_rois_b{budget}.pdf", bbox_inches="tight")


if __name__ == "__main__":

    EXPERIMENT_TAG = "revenue_rois"
    path_save = f"{PATH_SAVE}revenue_rois/"
    os.makedirs(path_save, exist_ok=True)

    config_learner = "soda1_revenue.yaml"
    n_bidder = 2

    for budget in [1, 2]:

        # plot revenue
        plot_revenue(budget, PATH_TO_CONFIGS, PATH_TO_EXPERIMENTS, path_save)

        # plot strategies
        # for payment_rule in ["fp", "sp"]:
        #    plot_strategies_revenue(
        #        budget,
        #        payment_rule,
        #        PATH_TO_CONFIGS,
        #        PATH_TO_EXPERIMENTS,
        #        path_save,
        #    )

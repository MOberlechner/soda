import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.ad_auctions.config_exp import *
from projects.ad_auctions.evaluation.util import *
from soda.game import Game
from soda.strategy import Strategy
from soda.util.evaluation import get_results


def plot_strategies_revenue(
    config_games,
    labels,
    payment_rule,
    path_to_configs,
    path_to_experiments,
    path_save,
    tag: str = "",
):
    os.makedirs(path_save, exist_ok=True)
    experiment_tag = "revenue"

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
            experiment_tag,
            run=0,
        )
        x, bids_mean, bids_std, bne = get_bids(game, strategies, agent="1")

        # plot results
        if bne is not None:
            ax.plot(x, bne, color="k", linestyle="--")
        ax.plot(x, bids_mean, color=color, linestyle="-", label=label, linewidth=2)
        ax.fill_between(
            x, bids_mean - bids_std, bids_mean + bids_std, alpha=0.4, color=color
        )

    ax.legend(fontsize=PARAM["fontsize_legend"], loc=2)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    fig.savefig(
        f"{path_save}/revenue_strat_{tag}{payment_rule}_{game.n_bidder}.pdf",
        bbox_inches="tight",
    )


def plot_revenue(
    n_bidder: int, path_to_configs, path_to_experiments, path_save, tag: str = ""
):

    labels = ["QL", "ROI", "ROSB"]
    learner = "soda1_revenue"

    df = get_revenue(path_to_experiments, "revenue")
    revenue = {}
    for payment_rule in ["fp", "sp"]:
        settings = [
            f"{tag}ql_{payment_rule}_{n_bidder}",
            f"{tag}roi_{payment_rule}_{n_bidder}",
            f"{tag}rosb_{payment_rule}_{n_bidder}",
        ]
        revenue[payment_rule] = [
            df.loc[(df.setting == s) & (df.learner == learner), "mean"].item()
            for s in settings
        ]

    fig, ax = set_axis("Utility Model", "Revenue")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.55)
    # plot revenue
    x = np.linspace(0.2, 0.8, 3)
    ax.bar(x - 0.055, revenue["fp"], width=0.1, color=COLORS)
    ax.bar(x + 0.055, revenue["sp"], width=0.1, color=COLORS, alpha=0.5)
    ax.bar(
        x + 0.055,
        revenue["sp"],
        width=0.1,
        facecolor="none",
        hatch="///",
        edgecolor=COLORS,
    )
    ax.set_xticks(np.linspace(0.2, 0.8, 3), labels)
    # legend
    ax.bar([-1], [1], label="First-Price", facecolor="white", edgecolor="k")
    ax.bar([-1], [1], label="Second-Price", hatch="//", color="white", edgecolor="k")
    ax.legend(fontsize=PARAM["fontsize_legend"], loc=2)
    fig.savefig(f"{path_save}/revenue_{tag}{n_bidder}.pdf", bbox_inches="tight")


if __name__ == "__main__":

    EXPERIMENT_TAG = "revenue"
    path_save = f"{PATH_SAVE}/{EXPERIMENT_TAG}"
    os.makedirs(PATH_SAVE, exist_ok=True)

    # 2 agents uniform prior
    for payment_rule in ["fp", "sp"]:
        config_learner = "soda1_revenue.yaml"
        config_games = [
            f"revenue/ql_{payment_rule}_2.yaml",
            f"revenue/roi_{payment_rule}_2.yaml",
            f"revenue/rosb_{payment_rule}_2.yaml",
        ]
        labels = ["QL", "ROI", "ROSB"]
        plot_strategies_revenue(
            config_games,
            labels,
            payment_rule,
            PATH_TO_CONFIGS,
            PATH_TO_EXPERIMENTS,
            path_save,
        )
    n_bidder = 2
    plot_revenue(n_bidder, PATH_TO_CONFIGS, PATH_TO_EXPERIMENTS, path_save)

    # 2 agents gaussian (truncated) prior
    for payment_rule in ["fp", "sp"]:
        config_learner = "soda1_revenue.yaml"
        config_games = [
            f"revenue/gaus_ql_{payment_rule}_2.yaml",
            f"revenue/gaus_roi_{payment_rule}_2.yaml",
            f"revenue/gaus_rosb_{payment_rule}_2.yaml",
        ]
        labels = ["QL", "ROI", "ROSB"]
        plot_strategies_revenue(
            config_games,
            labels,
            payment_rule,
            PATH_TO_CONFIGS,
            PATH_TO_EXPERIMENTS,
            path_save,
            tag="gaus_",
        )
    n_bidder = 2
    plot_revenue(n_bidder, PATH_TO_CONFIGS, PATH_TO_EXPERIMENTS, path_save, tag="gaus_")

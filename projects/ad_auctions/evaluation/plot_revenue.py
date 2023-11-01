import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.ad_auctions.config_exp import *
from projects.util import get_results
from soda.game import Game
from soda.strategy import Strategy

PARAM = {
    "fontsize_title": 14,
    "fontsize_legend": 13,
    "fontsize_label": 13,
}
COLORS = ["#003f5c", "#ffa600", "#bc5090"]


def set_axis(xlabel: str, ylabel: str, dpi=100, figsize=(5, 5)):
    """General settings for axis"""
    fig = plt.figure(tight_layout=True, dpi=dpi, figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel, fontsize=PARAM["fontsize_label"])
    ax.set_ylabel(ylabel, fontsize=PARAM["fontsize_label"])
    ax.grid(
        linestyle="-",
        linewidth=0.25,
        color="lightgrey",
        zorder=-10,
        alpha=0.5,
        axis="y",
    )
    return fig, ax


def get_bids(game: Game, strategies: Dict[str, Strategy], agent: str):
    x = game.o_discr[agent]
    bids = [strategies[agent].sample_bids(obs * np.ones(100)) for obs in x]
    bids_mean, bids_std = np.mean(bids, axis=1), np.std(bids, axis=1)
    bne = game.mechanism.get_bne(agent, x)
    return x, bids_mean, bids_std, bne


def get_revenue(path_to_experiments):
    df = pd.read_csv(os.path.join(path_to_experiments, "log/revenue/log_sim_agg.csv"))
    return df[df.metric == "revenue"].reset_index(drop=True)


def plot_strategies_revenue(
    config_games,
    labels,
    payment_rule,
    path_to_configs,
    path_to_experiments,
    path_save,
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
        f"{path_save}/strategies_revenue_{payment_rule}_{game.n_bidder}.pdf",
        bbox_inches="tight",
    )


def plot_revenue(n_bidder: int, path_to_configs, path_to_experiments, path_save):

    labels = ["QL", "ROI", "ROSB"]
    learner = "soda1_revenue"

    df = get_revenue(path_to_experiments)
    revenue = {}
    for payment_rule in ["fp", "sp"]:
        settings = [
            f"ql_{payment_rule}_{n_bidder}",
            f"roi_{payment_rule}_{n_bidder}",
            f"rosb_{payment_rule}_{n_bidder}",
        ]
        revenue[payment_rule] = [
            df.loc[(df.setting == s) & (df.learner == learner), "mean"].item()
            for s in settings
        ]

    fig, ax = set_axis("Utility Model", "Revenue")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_bidder / 4)
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
    fig.savefig(f"{path_save}/revenue_{n_bidder}.pdf", bbox_inches="tight")


if __name__ == "__main__":

    EXPERIMENT_TAG = "revenue"
    PATH_SAVE = "projects/ad_auctions/results/"
    os.makedirs(PATH_SAVE, exist_ok=True)

    for payment_rule in ["fp", "sp"]:
        config_learner = "soda1_revenue.yaml"
        config_games = [
            f"ql_{payment_rule}_2.yaml",
            f"roi_{payment_rule}_2.yaml",
            f"rosb_{payment_rule}_2.yaml",
        ]
        labels = ["QL", "ROI", "ROSB"]
        plot_strategies_revenue(
            config_games,
            labels,
            payment_rule,
            PATH_TO_CONFIGS,
            PATH_TO_EXPERIMENTS,
            PATH_SAVE,
        )

    plot_revenue(2, PATH_TO_CONFIGS, PATH_TO_EXPERIMENTS, PATH_SAVE)

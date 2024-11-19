import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.contests.config_exp import *
from soda.game import Game
from soda.strategy import Strategy
from soda.util.evaluation import get_results


def set_axis(xlabel: str, ylabel: str, dpi=100, figsize=(6, 5)):
    """General settings for axis"""
    fig = plt.figure(tight_layout=True, dpi=dpi, figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
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
    return x, bids_mean, bids_std


def plot_crowdsourcing(
    experiment_tag,
    config_games,
    config_learner,
    labels,
):
    # fig, axis
    fig, ax = set_axis("valuation (type) v", "effort e")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 0.75)

    # legend BNE
    ax.plot([], [], color="k", linestyle="--", label="BNE")

    for i, config_game in enumerate(config_games):
        # get data from computed strategies
        game, _, strategies = get_results(
            config_game,
            config_learner,
            PATH_TO_CONFIGS,
            PATH_TO_EXPERIMENTS,
            experiment_tag=experiment_tag,
            run=0,
        )
        agent = game.bidder[0]

        # bids
        x, bids_mean, bids_std = get_bids(game, strategies, agent=agent)
        ax.plot(
            x, bids_mean, color=COLORS[i], linestyle="-", label=labels[i], linewidth=2
        )
        ax.fill_between(
            x, bids_mean - bids_std, bids_mean + bids_std, alpha=0.4, color=COLORS[i]
        )

        # bne
        x = game.o_discr[agent]
        bne = game.mechanism.get_bne(agent, x)
        ax.plot(x, bne, color="k", linestyle="--")

    ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)
    fig.savefig(
        f"{PATH_TO_RESULTS}/{experiment_tag}.pdf",
        bbox_inches="tight",
    )


def plot_tullock(
    experiment_tag,
    config_games,
    config_learner,
    labels,
    ylim,
    save_tag: str = "",
):
    # fig, axis
    fig, ax = set_axis("valuation (type) v", "effort e")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(ylim)

    for i, config_game in enumerate(config_games):
        # get data from computed strategies
        game, _, strategies = get_results(
            config_game,
            config_learner,
            PATH_TO_CONFIGS,
            PATH_TO_EXPERIMENTS,
            experiment_tag=experiment_tag,
            run=0,
        )
        agent = game.bidder[0]

        # bids
        x, bids_mean, bids_std = get_bids(game, strategies, agent=agent)
        ax.plot(
            x, bids_mean, color=COLORS[i], linestyle="-", label=labels[i], linewidth=2
        )
        ax.fill_between(
            x, bids_mean - bids_std, bids_mean + bids_std, alpha=0.4, color=COLORS[i]
        )

    ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)

    fig.savefig(
        f"{PATH_TO_RESULTS}/{experiment_tag}{save_tag}.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    # Plot Crowdsourcing
    experiment_tag = "crowdsourcing"
    os.makedirs(PATH_TO_RESULTS, exist_ok=True)

    config_learner = "soda2_eta10_beta05.yaml"
    config_games = [
        "crowdsourcing/bidder3_price1.yaml",
        "crowdsourcing/bidder3_price2.yaml",
        "crowdsourcing/bidder3_price3.yaml",
    ]
    labels = [r"$w_1=1.0$", r"$w_1=0.8$", r"$w_1=0.6$"]
    plot_crowdsourcing(experiment_tag, config_games, config_learner, labels)

    # Plot Tullock Symmetric
    experiment_tag = "tullock"
    os.makedirs(PATH_TO_RESULTS, exist_ok=True)

    config_learner = "soda2_eta10_beta05.yaml"
    config_games = [
        "tullock/bidder2_csf1.yaml",
        "tullock/bidder2_csf2.yaml",
        "tullock/bidder2_csf3.yaml",
        "tullock/bidder2_csf4.yaml",
    ]
    labels = [
        r"$\varepsilon=0.5$",
        r"$\varepsilon=1.0$",
        r"$\varepsilon=2.0$",
        r"$\varepsilon=5.0$",
    ]
    plot_tullock(
        experiment_tag,
        config_games,
        config_learner,
        labels,
        ylim=(-0.02, 0.45),
        save_tag="_sym",
    )

    # Plot Tullock Asymmetric
    experiment_tag = "tullock"
    os.makedirs(PATH_TO_RESULTS, exist_ok=True)

    config_learner = "soda2_eta10_beta05.yaml"
    config_games = [
        "tullock/bidder2_asym1.yaml",
        "tullock/bidder2_asym2.yaml",
        "tullock/bidder2_asym3.yaml",
        "tullock/bidder2_asym4.yaml",
    ]
    labels = [
        r"$v_2 \sim U([0,1])$",
        r"$v_2 \sim U([1,2])$",
        r"$v_2 \sim U([2,3])$",
        r"$v_2 \sim U([3,4])$",
    ]
    plot_tullock(
        experiment_tag,
        config_games,
        config_learner,
        labels,
        ylim=(-0.02, 0.30),
        save_tag="_asym",
    )

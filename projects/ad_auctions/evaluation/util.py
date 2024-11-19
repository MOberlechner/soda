import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from soda.game import Game
from soda.strategy import Strategy

PARAM = {
    "fontsize_title": 16,
    "fontsize_legend": 15,
    "fontsize_label": 15,
}
COLORS = ["#003f5c", "#ffa600", "#bc5090"]


def set_axis(xlabel: str, ylabel: str, dpi=100, figsize=(5, 5)):
    """General settings for axis"""
    fig = plt.figure(tight_layout=True, dpi=dpi, figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel, fontsize=PARAM["fontsize_label"])
    ax.set_ylabel(ylabel, fontsize=PARAM["fontsize_label"])
    ax.tick_params(axis="x", labelsize=PARAM["fontsize_label"] - 1)
    ax.tick_params(axis="y", labelsize=PARAM["fontsize_label"] - 1)
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


def get_revenue(path_to_experiments, experiment_tag):
    df = pd.read_csv(
        os.path.join(path_to_experiments, f"{experiment_tag}/log/simulation_aggr.csv")
    )
    return df[(df.metric == "revenue") & (df.agent == "mechanism")].reset_index(
        drop=True
    )

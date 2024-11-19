import os
from typing import Dict

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.soda.config_exp import *
from projects.soda.evaluation.generate_plots import get_data, set_axis
from soda.game import Game
from soda.strategy import Strategy
from soda.util.config import Config
from soda.util.evaluation import get_runtimes


def get_bne_fpsb(n: int, odd: bool = True) -> np.ndarray:
    """Discrete BNE for FPSB unform prior, 2 bidder

    Args:
        n (int): number of discretization points
        odd (bool, optional): which BNE. Defaults to True.

    Returns:
        np.ndarray: bne
    """
    # bne_odd
    if odd:
        bne = np.vstack(
            [
                [0] * (n // 2 - 1),
                (np.kron(np.eye(n // 2 - 1), np.ones(2)).T),
                [0] * (n // 2 - 1),
            ]
        )
        bne = np.hstack(
            [np.eye(1, n, 0).reshape(n, 1), bne, np.eye(1, n, n - 1).reshape(n, 1)]
        )
        x = np.hstack([bne, np.zeros((n, n // 2 - 1))])

    else:
        # bne even
        x = np.hstack([np.kron(np.eye(n // 2), np.ones(2)).T, np.zeros((n, n // 2))])
    return x / x.sum()


def generate_plots_discretization():
    """Generate Plots for Figure 8 (Runtime)"""
    df = get_runtimes(PATH_TO_EXPERIMENTS, "discretization")
    discr = [16, 32, 64, 128]

    # General Formulation
    runtime2 = [
        df[df.setting == f"general_2_discr{d:03}"].time_total_mean.item() for d in discr
    ]
    runtime3 = [
        df[df.setting == f"general_3_discr{d:03}"].time_total_mean.item() for d in discr
    ]

    fig, ax = set_axis(
        (12, 132),
        (0.05, 350),
        "Runtime - General Formulation",
        "Discretization",
        "Runtime in s",
    )
    ax.set_xticks(discr)
    ax.set_yscale("log")
    ax.set_aspect(0.6 * 120 / (np.log10(350) - np.log10(0.05)))
    ax.plot(discr, runtime2, color=COLORS[0], marker=MARKER[0], label="n=2")
    ax.plot(discr, runtime3, color=COLORS[1], marker=MARKER[1], label="n=3")
    ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)
    path_save = os.path.join(PATH_TO_EXPERIMENTS, "plots", f"figure_8_1")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")

    # Symmeetric Formulation
    runtime2 = [
        df[df.setting == f"fast_2_discr{d:03}"].time_total_mean.item() for d in discr
    ]
    runtime10 = [
        df[df.setting == f"fast_10_discr{d:03}"].time_total_mean.item() for d in discr
    ]

    fig, ax = set_axis(
        (12, 132),
        (0.05, 350),
        "Runtime - Symmetric Formulation",
        "Discretization",
        "Runtime in s",
    )
    ax.set_xticks(discr)
    ax.set_yscale("log")
    ax.set_aspect(0.6 * 120 / (np.log10(350) - np.log10(0.05)))
    ax.plot(discr, runtime2, color=COLORS[0], marker=MARKER[0], label="n=2")
    ax.plot(discr, runtime10, color=COLORS[1], marker=MARKER[1], label="n=10")
    ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)
    path_save = os.path.join(PATH_TO_EXPERIMENTS, "plots", f"figure_8_2")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


def generate_plots_vs():
    """Generate Plot for Figure 13 (Variational Stability)"""
    cmap_greys = matplotlib.colors.ListedColormap(["none", "black"])
    cmap_purple = matplotlib.colors.ListedColormap(["none", COLORS[1]])

    # create setting
    config_game = os.path.join(PATH_TO_CONFIGS, "game", "example/fpsb_2_discr032.yaml")
    config_learner = os.path.join(PATH_TO_CONFIGS, "learner", "soda1_eta10_beta05.yaml")
    config = Config(config_game, config_learner)
    game, learner = config.create_setting()
    strategies = config.create_strategies(game)

    # Plot BNE with Best Response
    for i in range(2):
        if i == 0:
            strategies["1"].x = get_bne_fpsb(game.n, True)
            grad = game.mechanism.compute_gradient(game, strategies, "1")
            br = [np.where(g == g.max())[0].min() for g in grad]
        else:
            strategies["1"].x = get_bne_fpsb(game.n, False)
            grad = game.mechanism.compute_gradient(game, strategies, "1")
            br = [np.where(g == g.max())[0].max() for g in grad]

        fig, ax = set_axis((-1 / 64, 65 / 64), (-1 / 64, 65 / 64), f"BNE {i+1}")
        ax.imshow(
            strategies["1"].x.T,
            extent=(-1 / 64, 65 / 64, -1 / 64, 65 / 64),
            origin="lower",
            vmin=0,
            cmap=cmap_greys,
            zorder=2,
        )
        ax.scatter(
            strategies["1"].o_discr,
            strategies["1"].a_discr[br],
            marker="s",
            s=55,
            edgecolors=COLORS[2],
            c="none",
            linewidths=2,
            zorder=3,
        )
        ax.scatter([], [], marker="s", s=110, c="k", label=r"BNE $s^*$")
        ax.scatter(
            [],
            [],
            marker="s",
            s=110,
            c="none",
            edgecolors=COLORS[2],
            label=r"Best Response $s^{br}$",
            linewidths=2,
        )
        ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)
        path_save = os.path.join(PATH_TO_RESULTS, "plots", f"figure_13_{i+1}")
        fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")

    # Plot Counterexample VS (Collusive strategy)
    c = np.zeros((game.n, game.m))
    for i in range(game.n):
        c[i, i // 4] = 1 / game.n

    fig, ax = set_axis((-1 / 64, 65 / 64), (-1 / 64, 65 / 64), "Collusive Strategy")
    ax.imshow(
        get_bne_fpsb(game.n, True).T,
        extent=(-1 / 64, 65 / 64, -1 / 64, 65 / 64),
        origin="lower",
        vmin=0,
        cmap=cmap_greys,
        zorder=2,
    )
    ax.imshow(
        c.T,
        extent=(-1 / 64, 65 / 64, -1 / 64, 65 / 64),
        origin="lower",
        vmin=0,
        cmap=cmap_purple,
        zorder=2,
    )
    ax.scatter([], [], marker="s", s=100, c="k", label=r"BNE 1 $s^*$")
    ax.scatter(
        [],
        [],
        marker="s",
        s=100,
        c=COLORS[1],
        label=r"Collusive Strategy $s^c$",
        linewidths=2,
    )
    ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)
    path_save = os.path.join(PATH_TO_RESULTS, "plots", f"figure_13_3")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS, "plots"), exist_ok=True)
    generate_plots_vs()
    generate_plots_discretization()

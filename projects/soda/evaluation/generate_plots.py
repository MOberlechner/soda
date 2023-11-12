import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.soda.config_exp import *
from soda.game import Game
from soda.strategy import Strategy
from soda.util.evaluation import get_results


def get_bids(game: Game, strategies: Dict[str, Strategy], agent: str):
    """Sample observation from mechanism and corresponding bids from agent's strategy"""
    idx_agent = game.bidder.index(agent)
    obs = game.mechanism.sample_types(SAMPLES)[idx_agent]
    bids = strategies[agent].sample_bids(obs)
    return obs, bids


def get_bne(game: Game, agent: str):
    """Get BNE for agent in mechanism"""
    lb, up = game.mechanism.o_space[agent]
    x = np.linspace(lb, up, 100)
    bne = game.mechanism.get_bne(agent, x)
    return x, bne


def get_data(config_game, config_learner, experiment_tag, agent, run=0):
    game, _, strategies = get_results(
        config_game,
        config_learner,
        PATH_TO_CONFIGS,
        PATH_TO_EXPERIMENTS,
        experiment_tag,
        run=0,
    )
    obs, bids = get_bids(game, strategies, agent)
    x, bne = get_bne(game, agent)
    return obs, bids, x, bne


def set_axis(xlim, ylim, title, xlabel: str = "Observation o", ylabel: str = "Bid b"):
    """General settings for axis"""
    fig = plt.figure(tight_layout=True, dpi=DPI)
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    ax.grid(linestyle="-", linewidth=0.25, color="lightgrey", zorder=-10, alpha=0.5)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_title(title, fontsize=FONTSIZE_TITLE)
    ax.set_aspect("equal")
    return fig, ax


def generate_plots_interdependent():
    """Generate plots for Figure 2  (Affiliated & Common Value Model)"""
    configs_game = [
        "interdependent/affiliated_values.yaml",
        "interdependent/common_value.yaml",
    ]
    labels = ["Affiliated Values Model", "Common Value Model"]

    for i in range(2):
        # affiliated values model
        obs, bids, x, bne = get_data(
            configs_game[i], "soda1_eta100_beta50.yaml", "interdependent", "1"
        )
        fig, ax = set_axis((0, 2), (0, 1.5), labels[i])
        ax.scatter(
            obs,
            bids,
            facecolors=COLORS[0],
            edgecolors="none",
            marker=MARKER[0],
            s=MARKER_SIZE,
            zorder=2,
            alpha=1,
        )
        ax.plot(x, bne, color=COLORS[0], linestyle="-", zorder=1, alpha=0.9)
        # legend
        ax.scatter(
            [],
            [],
            facecolors=COLORS[0],
            edgecolors="none",
            marker=MARKER[0],
            s=40,
            label="$\mathrm{SODA}_1$",
        )
        ax.plot(
            [], [], label="analyt. BNE", color=COLORS[0], linestyle="-", linewidth=2
        )
        ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)
        path_save = os.path.join(PATH_TO_RESULTS, "plots", f"figure2_{i+1}.pdf")
        fig.savefig(path_save, bbox_inches="tight")


def generate_plots_llg():
    """Generate plots for Figure 3 (LLG Auction)"""
    labels = ["Nearest-Bid Rule", "Nearest-Zero Rule", "Nearest-VCG Rule"]
    pr = ["nb", "nz", "nvcg"]
    gammas = [0.1, 0.5, 0.9]

    for j in range(3):
        fig, ax = set_axis((0, 1), (0, 1), labels[j])
        for i in range(3):
            obs, bids, x, bne = get_data(
                f"llg/{pr[j]}_gamma{i+1}.yaml", "soda2_eta50_beta05.yaml", "llg", "L"
            )
            # plot strategy & BNE
            ax.scatter(
                obs,
                bids,
                facecolors=COLORS[i],
                edgecolors="none",
                marker=MARKER[i],
                s=MARKER_SIZE,
                zorder=2,
                alpha=1,
            )
            ax.plot(x, bne, color=COLORS[i], linestyle="-", zorder=1)
            # legend
            ax.plot(
                [],
                [],
                color=COLORS[i],
                marker=MARKER[i],
                linestyle="-",
                linewidth=2,
                zorder=1,
                label=f"$\gamma={gammas[i]}$",
            )
        ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)
        path_save = os.path.join(PATH_TO_RESULTS, "plots", f"figure3_{j+1}.pdf")
        fig.savefig(path_save, bbox_inches="tight")


def generate_plots_llg_fp():
    """Generate plots for Figure 4 (LLG Auction First-Price)"""
    gammas = [0.1, 0.5, 0.9]
    for j in range(3):
        fig, ax = set_axis((0, 2), (0, 1), f"Correlation $\gamma = {gammas[j]}$")
        ax.set_aspect(2)
        for i, agent in enumerate(["L", "G"]):
            obs, bids, x, bne = get_data(
                f"llg/fp_gamma{j+1}.yaml",
                "sofw.yaml",
                "llg",
                agent,
                run=1,
            )
            # plot strategy & BNE
            ax.scatter(
                obs,
                bids,
                facecolors=COLORS[i],
                edgecolors="none",
                marker=MARKER[i],
                s=MARKER_SIZE,
                zorder=2,
                alpha=1,
            )
            # legend
            ax.scatter(
                [],
                [],
                facecolors=COLORS[i],
                edgecolors="none",
                marker=MARKER[0],
                s=40,
                label={"L": "Local Bidder", "G": "Global Bidder"}[agent],
            )
        ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)
        path_save = os.path.join(PATH_TO_RESULTS, "plots", f"figure4_{j+1}.pdf")
        fig.savefig(path_save, bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS, "plots"), exist_ok=True)
    generate_plots_interdependent()
    generate_plots_llg()
    generate_plots_llg_fp()

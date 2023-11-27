import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.soda.config_exp import *
from soda.game import Game
from soda.strategy import Strategy
from soda.util.config import Config
from soda.util.evaluation import (
    aggregate_metrics_over_runs,
    get_bids,
    get_bne,
    get_log_files,
    get_results,
)


def get_data(config_game, config_learner, experiment_tag, agent, run=0):
    game, _, strategies = get_results(
        config_game,
        config_learner,
        PATH_TO_CONFIGS,
        PATH_TO_EXPERIMENTS,
        experiment_tag,
        run=0,
    )
    obs, bids = get_bids(game, strategies, agent, SAMPLES)
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


def plot_scatter(ax, obs, bids, index, label_legend: str = None):
    """Standard scatter plot we use to plot obs/bids"""
    ax.scatter(
        obs,
        bids,
        facecolors=COLORS[index],
        edgecolors="none",
        marker=MARKER[index],
        s=MARKER_SIZE,
        zorder=2,
    )
    if label_legend is not None:
        ax.scatter(
            [],
            [],
            facecolors=COLORS[index],
            edgecolors="none",
            marker=MARKER[index],
            s=40,
            label=label_legend,
        )
    return ax


def generate_plots_example():
    """Generate plots for Figure 1 (Example FPSB)"""
    # create setting
    config_game = os.path.join(PATH_TO_CONFIGS, "game", "example/fpsb_2_discr020.yaml")
    config_learner = os.path.join(PATH_TO_CONFIGS, "learner", "soda1_eta20_beta05.yaml")
    config = Config(config_game, config_learner)
    game, learner = config.create_setting()
    strategies = config.create_strategies(game)

    # import computed strategies
    learner_name = os.path.basename(config_learner).replace(".yaml", "")
    game_name = os.path.basename(config_game).replace(".yaml", "")
    name = f"{learner_name}_{game_name}_run_0"
    path = os.path.join(PATH_TO_EXPERIMENTS, "strategies", "example")

    # initial strategy
    strategies["1"].load(name, path, load_init=True)
    fig, ax = set_axis((0, 1), (0, 1), f"Random Initial Strategy")
    ax.imshow(
        strategies["1"].x.T,
        extent=(0, 1, 0, 1),
        origin="lower",
        vmin=0,
        cmap="Greys",
        zorder=2,
    )
    path_save = os.path.join(PATH_TO_EXPERIMENTS, "plots", f"figure_1_1")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")

    # computed strategy
    strategies["1"].load(name, path, load_init=False)
    fig, ax = set_axis((0, 1), (0, 1), f"Computed Strategy")
    ax.imshow(
        strategies["1"].x.T,
        extent=(0, 1, 0, 1),
        origin="lower",
        vmin=0,
        cmap="Greys",
        zorder=2,
    )
    path_save = os.path.join(PATH_TO_EXPERIMENTS, "plots", f"figure_1_2")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")

    # sampled bids
    obs, bids = get_bids(game, strategies, "1", 150)
    bids_bne = game.mechanism.get_bne("1", obs)
    fig, ax = set_axis((-0.01, 1.01), (-0.01, 1.01), f"Sampled Bids")
    plot_scatter(ax, obs, bids_bne, 1, label_legend="analyt. BNE")
    plot_scatter(ax, obs, bids, 0, label_legend=r"$SODA_1$")
    ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)
    path_save = os.path.join(PATH_TO_EXPERIMENTS, "plots", f"figure_1_3")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


def generate_plots_interdependent():
    """Generate plots for Figure 2  (Affiliated & Common Value Model)"""
    configs_game = [
        "interdependent/common_value.yaml",
        "interdependent/affiliated_values.yaml",
    ]
    labels = ["Common Value Model", "Affiliated Values Model"]

    for i in range(2):
        # affiliated values model
        obs, bids, x, bne = get_data(
            configs_game[i], "soda1_eta100_beta50.yaml", "interdependent", "1"
        )
        fig, ax = set_axis((0, 2), (0, 1.5), labels[i])

        # plot strategy & BNE
        ax = plot_scatter(ax, obs, bids, index=0, label_legend="$\mathrm{SODA}_1$")
        ax.plot(x, bne, color=COLORS[0], linestyle="-", zorder=1, alpha=0.9)

        # legend
        ax.plot(
            [], [], label="analyt. BNE", color=COLORS[0], linestyle="-", linewidth=2
        )
        ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)
        path_save = os.path.join(PATH_TO_EXPERIMENTS, "plots", f"figure_2_{i+1}")
        fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


def generate_plots_llg():
    """Generate plots for Figure 3 (LLG Auction)"""
    labels = ["Nearest-Zero Rule", "Nearest-VCG Rule", "Nearest-Bid Rule"]
    pr = ["nz", "nvcg", "nb"]
    gammas = [0.1, 0.5, 0.9]

    for j in range(3):
        fig, ax = set_axis((0, 1), (0, 1), labels[j])
        for i in range(3):
            obs, bids, x, bne = get_data(
                f"llg/{pr[j]}_gamma{i+1}.yaml", "soda2_eta50_beta05.yaml", "llg", "L"
            )
            # plot strategy & BNE
            ax = plot_scatter(ax, obs, bids, index=i, label_legend=None)
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
        path_save = os.path.join(PATH_TO_EXPERIMENTS, "plots", f"figure_3_{j+1}")
        fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


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
            # plot strategy
            label_legend = {"L": "Local Bidder", "G": "Global Bidder"}[agent]
            ax = plot_scatter(ax, obs, bids, i, label_legend)

        ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)
        path_save = os.path.join(PATH_TO_EXPERIMENTS, "plots", f"figure_4_{j+1}")
        fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


def generate_plots_split_award():
    """Generate plots for Figure 5 (Split-Award Gaussian)"""
    game, _, strategies = get_results(
        "split_award/sa_gaussian.yaml",
        "sofw.yaml",
        PATH_TO_CONFIGS,
        PATH_TO_EXPERIMENTS,
        "split_award",
        run=0,
    )
    # get bids
    agent = "1"
    obs, bids = get_bids(game, strategies, agent, SAMPLES)

    # get BNES
    x, bne_pool_upper = get_bne(game, agent)
    bne_pool_lower = game.mechanism.equilibrium_pooling(agent, x, "lower")
    bne_wta = game.mechanism.equilibrium_wta(agent, x)

    labels = ["Sole Source Award (100%)", "Split Award (50%)"]
    for i in range(2):
        fig, ax = set_axis((1, 1.4), (0, 2.5), labels[i])
        ax.set_aspect(0.75 * 0.4 / 2.5)

        # plot strategy and BNE
        ax = plot_scatter(ax, obs, bids[i], index=0, label_legend="$\mathrm{SODA}_1$")
        ax.fill_between(
            x,
            bne_pool_lower[i],
            bne_pool_upper[i],
            color=COLORS[0],
            zorder=1,
            alpha=0.3,
        )
        ax.plot(x, bne_wta[i], linestyle="--", color="k")

        # legend
        ax.plot([], [], label="WTA-BNE", color="k", linestyle="--", linewidth=2)
        ax.fill_between(
            [], [], color=COLORS[0], zorder=1, alpha=0.3, label="Pooling-BNE"
        )
        ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)

        path_save = os.path.join(PATH_TO_EXPERIMENTS, "plots", f"figure_5_{i+1}")
        fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


def generate_plots_risk():
    """Generate Plots for Figure 6 (risk)"""
    risk = [0.5, 0.7, 0.9]
    # First-Price
    fig, ax = set_axis((0, 1), (0, 0.8), "First-Price Auction")
    ax.set_aspect(1.25)
    for i, r in enumerate(risk):
        obs, bids, x, bne = get_data(
            f"risk/fpsb_risk{2*i}.yaml",
            "soda1_eta20_beta05.yaml",
            "risk",
            "1",
        )
        # plot strategy
        ax = plot_scatter(ax, obs, bids, i, r"$\rho$" + f" = {r}")
        ax.plot(x, bne, color=COLORS[i], linestyle="-", zorder=1)
    ax.plot(x, 0.5 * x, color="k", linestyle="--", label="risk-neutral")
    ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)
    path_save = os.path.join(PATH_TO_EXPERIMENTS, "plots", f"figure_6_1")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")

    # All-Pay
    fig, ax = set_axis((0, 1), (0, 0.8), "All-Pay Auction")
    ax.set_aspect(1.25)
    for i, r in enumerate(risk):
        obs, bids, x, bne = get_data(
            f"risk/allpay_risk{2*i}.yaml",
            "soda1_eta25_beta05.yaml",
            "risk",
            "1",
        )
        # plot strategy
        ax = plot_scatter(ax, obs, bids, i, r"$\rho$" + f" = {r}")
    ax.plot(x, 0.5 * x**2, color="k", linestyle="--", label="risk-neutral")
    ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)
    path_save = os.path.join(PATH_TO_EXPERIMENTS, "plots", f"figure_6_2")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")

    # Revenue
    risk = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    _, df_sim = get_log_files(PATH_TO_EXPERIMENTS, "risk")
    cols_index = ["mechanism", "setting", "learner", "agent"]
    df = aggregate_metrics_over_runs(df_sim, cols_index, ["revenue"])
    df = df.sort_values(["mechanism", "setting"]).reset_index(drop=True)

    fig, ax = set_axis(
        (0.45, 1.05), (0.325, 0.575), "Revenue", r"Risk Parameter $\rho$", "Revenue"
    )
    ax.set_aspect(0.6 / 0.25)
    # First-Price
    revenue = df[
        (df.mechanism == "single_item") & (df.learner == "soda1_eta20_beta05")
    ].mean_revenue
    ax.plot(
        risk,
        revenue,
        color=COLORS[0],
        marker=MARKER[0],
        linestyle="-",
        linewidth=2,
        zorder=1,
        label="First-Price",
    )
    # All-Pay
    revenue = df[
        (df.mechanism == "all_pay") & (df.learner == "soda1_eta25_beta05")
    ].mean_revenue
    ax.plot(
        risk,
        revenue,
        color=COLORS[1],
        marker=MARKER[1],
        linestyle="-",
        linewidth=2,
        zorder=1,
        label="All-Pay",
    )
    ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)
    path_save = os.path.join(PATH_TO_EXPERIMENTS, "plots", f"figure_6_3")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


def generate_plots_contests():
    """Generate Plots for Figure 7 (contests)"""
    labels = ["Symmetric", "Asymmetric, Weak Bidder", "Asymmetric, Strong Bidder"]
    ratio = [0.5, 1.0, 1.5]

    # symmetric
    fig, ax = set_axis((0, 1), (0, 0.5), "Symmetric")
    ax.set_aspect(2)
    for i, r in enumerate(ratio):
        obs, bids, x, bne = get_data(
            f"contests/sym_ratio{i+1}.yaml",
            "soda1_eta100_beta05.yaml",
            "contests",
            "1",
        )
        # plot strategy
        ax = plot_scatter(ax, obs, bids, i, f"r = {r}")

    ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)
    path_save = os.path.join(PATH_TO_EXPERIMENTS, "plots", f"figure_7_1")
    fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")

    # asymmetric
    for j in range(2):
        agent, label = [("weak", "Weak Bidder"), ("strong", "Strong Bidder")][j]
        fig, ax = set_axis((0 + j, 1 + j), (0, 0.5), f"Asymmetric, {label}")
        ax.set_aspect(2)
        for i, r in enumerate(ratio):
            obs, bids, x, bne = get_data(
                f"contests/asym_ratio{i+1}.yaml",
                "soda1_eta100_beta05.yaml",
                "contests",
                agent,
            )
            # plot strategy
            ax = plot_scatter(ax, obs, bids, i, f"r = {r}")

        ax.legend(fontsize=FONTSIZE_LEGEND, loc=2)
        path_save = os.path.join(PATH_TO_EXPERIMENTS, "plots", f"figure_7_{j+2}")
        fig.savefig(f"{path_save}.{FORMAT}", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_EXPERIMENTS, "plots"), exist_ok=True)
    generate_plots_example()
    generate_plots_interdependent()
    generate_plots_llg()
    generate_plots_llg_fp()
    generate_plots_split_award()
    generate_plots_risk()
    generate_plots_contests()

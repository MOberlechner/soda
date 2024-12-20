# -------------------------------------------------------------------------------------------------------------------- #
#                                  SCRIPT WITH USEFUL METHODS FOR EVALUATION etc                                       #
# -------------------------------------------------------------------------------------------------------------------- #
import os
from typing import Dict

import numpy as np
import pandas as pd

from soda.game import Game
from soda.strategy import Strategy
from soda.util.config import Config


def get_results(
    config_game: dict,
    config_learner: dict,
    path_to_configs: str,
    path_to_experiment: str,
    experiment_tag: str,
    run: int = 0,
):
    """Import computed strategies for a given experiment

    Args:
        config_game (dict): config file for game
        config_learner (dict): config file for learner
        path_to_configs (str): path to config files
        path_to_experiment (str): path to experiments
        experiment_tag (str): experiment tag i.e. subdirectory in experiments
        run (int, optional): Run of experiment. Defaults to 0.
    """
    config_game = os.path.join(path_to_configs, "game", config_game)
    config_learner = os.path.join(path_to_configs, "learner", config_learner)

    config = Config(config_game, config_learner)
    game, learner = config.create_setting()
    strategies = config.create_strategies(game, init_method="nan")

    label_learner = os.path.basename(config_learner).replace(".yaml", "")
    label_setting = os.path.basename(config_game).replace(".yaml", "")
    name = f"{label_setting}_{label_learner}_run_{run}"
    filename = os.path.join(path_to_experiment, experiment_tag, "strategies", name)
    for i in strategies:
        strategies[i].load(filename, load_init=False)
    return game, learner, strategies


def get_bids(game: Game, strategies: Dict[str, Strategy], agent: str, sample_size: int):
    """Sample observation from mechanism and corresponding bids from agent's strategy"""
    idx_agent = game.bidder.index(agent)
    obs = game.mechanism.sample_types(sample_size)[idx_agent]
    bids = strategies[agent].sample_bids(obs)
    return obs, bids


def get_bne(game: Game, agent: str):
    """Get BNE for agent in mechanism"""
    lb, ub = game.mechanism.o_space[agent]
    x = np.linspace(lb + 1e-5, ub - 1e-5, 100)
    bne = game.mechanism.get_bne(agent, x)
    return x, bne


def create_table(
    path_to_experiments: str, experiment_tag: str, num_decimals: int = 3
) -> pd.DataFrame:
    """Create table with utility loss, l2 distance etc for experiment

    Args:
        path_to_experiments (str):
        experiment_tag (str):

    Returns:
        pd.DataFrame
    """
    cols_index = ["mechanism", "setting", "learner", "agent"]
    cols_metrics_sim = ["utility_loss_vs_bne", "l2_norm"]
    cols_metrics_com = ["utility_loss"]

    # get log files
    logs = get_log_files(path_to_experiments, experiment_tag)

    if logs["computation"] is None:
        print(f"computation log files for experiment '{experiment_tag}' are missing!")
        return None

    df_learn = aggregate_metrics_over_runs(
        logs["computation"], cols_index, cols_metrics_com
    )
    df_learn = create_str_col(
        df_learn, "util_loss_discr", "utility_loss", num_decimals=num_decimals
    )

    if logs["simulation"] is None:
        df = df_learn[cols_index + ["util_loss_discr"]]
    else:
        # add results from simulation to taoble
        df_sim = aggregate_metrics_over_runs(
            logs["simulation"], cols_index, cols_metrics_sim
        )
        df_sim = create_str_col(df_sim, "util_loss", "utility_loss_vs_bne")
        df_sim = create_str_col(df_sim, "l2_norm", "l2_norm")
        df = df_sim[cols_index + ["util_loss", "l2_norm"]].merge(
            df_learn[cols_index + ["util_loss_discr"]], on=cols_index, how="outer"
        )
    # merge runtime into table
    runtime = get_runtimes(path_to_experiments, experiment_tag)
    df = df.merge(runtime[cols_index + ["time"]], on=cols_index, how="outer")

    df.insert(
        2, "learner_label", [df["learner"][i].split("_")[0].upper() for i in df.index]
    )
    df = df.sort_values(
        by=["mechanism", "setting", "learner_label"],
        key=lambda x: x.map(custom_sort_learner),
    )
    return df.reset_index(drop=True).fillna("-")


def get_log_files(
    path_to_experiments: str, experiment_tag: str
) -> Dict[str, pd.DataFrame]:
    """Import aggregated log files from experiment"""
    sub_experiments = ["computation", "simulation", "evaluation"]
    cols_index = ["setting", "mechanism", "learner", "agent"]
    logs = []
    for sub_exp in sub_experiments:
        file_agg = os.path.join(
            path_to_experiments, experiment_tag, "log", f"{sub_exp}_aggr.csv"
        )
        if os.path.exists(file_agg):
            logs.append(pd.read_csv(file_agg, dtype={col: str for col in cols_index}))
        else:
            logs.append(None)
    return dict(zip(sub_experiments, logs))


def aggregate_metrics_over_runs(df: pd.DataFrame, cols_index: list, cols_metrics: list):
    """reformat table (long -> wide)"""
    df = df[df.metric.isin(cols_metrics)]
    df = df[~df["mean"].isna()]
    df = pd.pivot(df, index=cols_index, columns=["metric"], values=["mean", "std"])
    df.columns = ["_".join(c) for c in df.columns]
    return df.reset_index()


def get_runtimes(path_to_experiments: str, experiment_tag: str) -> pd.DataFrame:
    """Add column time (time_init + time_rum) to log_learn

    Args:
        path_to_experiments (str): path to respective subdirectory of experiments
        experiment_tag (str): experiment tag, i.e., subdirectory of log directory

    Returns:
        pd.DataFrame: df containing the runtimes
    """
    cols_index = ["setting", "mechanism", "learner", "agent"]
    # import file
    file_log_learn = os.path.join(
        path_to_experiments, experiment_tag, "log", "computation.csv"
    )
    df = pd.read_csv(file_log_learn, dtype={col: str for col in cols_index})
    # get runtimes (min, max)
    df["time_total"] = df["time_init"] + df["time_run"]
    df = df.groupby(cols_index).agg(
        {
            "time_init": "first",
            "time_run": ["min", "max", "mean"],
            "time_total": ["min", "max", "mean"],
        }
    )
    # create text for table
    df["time"] = [
        time_to_str(t_min=df["time_total"]["min"][i], t_max=df["time_total"]["max"][i])
        for i in df.index
    ]
    # get rid of multiindex
    df.columns = ["_".join(c) if c[1] != "" else c[0] for c in df.columns]
    return df.reset_index()


def time_to_str(t_min, t_max) -> str:
    """Method that brings runtime in format for table."""
    if t_max < 100:
        if t_min < 2:
            return f"{t_min:.1f}-{t_max:.1f} s"
        else:
            return f"{t_min:.0f}-{t_max:.0f} s"
    else:
        if t_min < 300:
            return f"{t_min/60:.1f}-{np.ceil(t_max/60):.1f} min"
        else:
            return f"{t_min/60:.0f}-{np.ceil(t_max/60):.0f} min"


def create_str_col(
    df: pd.DataFrame, new_column: str, column: str, num_decimals: int = 3
) -> pd.DataFrame:
    """add new column that contains 'mean (std)' in a single column"""
    df[new_column] = [
        metric_to_str(df[f"mean_{column}"][i], df[f"std_{column}"][i], num_decimals)
        for i in df.index
    ]
    return df


def metric_to_str(mean: float, std: float, num_decimals: int = 3) -> str:
    return f"{mean:.{num_decimals}f} ({std:.{num_decimals}f})"


def custom_sort_learner(val: str):
    learner_order = {"SODA1": 0, "SODA2": 1, "SOMA2": 2, "SOFW": 3, "FP": 4}
    if val in learner_order:
        return learner_order[val]
    else:
        return val

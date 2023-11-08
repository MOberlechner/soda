# -------------------------------------------------------------------------------------------------------------------- #
#                                  SCRIPT WITH USEFUL METHODS FOR EVALUATION etc                                       #
# -------------------------------------------------------------------------------------------------------------------- #
import os

import numpy as np
import pandas as pd

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
    strategies = config.create_strategies(game)

    learner_name = os.path.basename(config_learner).replace(".yaml", "")
    game_name = os.path.basename(config_game).replace(".yaml", "")
    name = f"{learner_name}_{game_name}_run_{run}"
    path = os.path.join(path_to_experiment, "strategies", experiment_tag)

    for i in strategies:
        strategies[i] = Strategy(i, game)
        strategies[i].load(name, path, load_init=False)

    return game, learner, strategies


def create_table(path_to_experiments: str, experiment_tag: str) -> pd.DataFrame:
    """Create table with utility loss, l2 distance etc for experiment

    Args:
        path_to_experiments (str):
        experiment_tag (str):

    Returns:
        pd.DataFrame
    """
    cols_index = ["mechanism", "setting", "learner", "agent"]
    cols_metrics_sim = ["l2_norm", "util_loss"]
    cols_metrics_learn = ["utility_loss"]

    # get log files
    df_learn, df_sim = get_log_files(path_to_experiments, experiment_tag)

    # get metrics over runs
    df_sim = aggregate_metrics_over_runs(df_sim, cols_index, cols_metrics_sim)
    df_learn = aggregate_metrics_over_runs(df_learn, cols_index, cols_metrics_learn)

    # create column with str
    df_learn = create_str_col(df_learn, "util_loss_discr", "utility_loss")
    df_sim = create_str_col(df_sim, "util_loss", "util_loss")
    df_sim = create_str_col(df_sim, "l2_norm", "l2_norm")

    # merge results and add runtimes
    df = df_sim[cols_index + cols_metrics_sim].merge(
        df_learn[cols_index + ["util_loss_discr"]], on=cols_index, how="outer"
    )
    runtime = get_runtimes(path_to_experiments, experiment_tag)
    df = df.merge(runtime[cols_index + ["time"]], on=cols_index, how="outer")

    return df.sort_values(cols_index).reset_index(drop=True).fillna("-")


def get_log_files(path_to_experiments: str, experiment_tag: str) -> pd.DataFrame:
    """Import log files from experiment"""
    file_log_learn_agg = os.path.join(
        path_to_experiments, "log", experiment_tag, "log_learn_agg.csv"
    )
    file_log_sim_agg = os.path.join(
        path_to_experiments, "log", experiment_tag, "log_sim_agg.csv"
    )
    df_learn = pd.read_csv(file_log_learn_agg)
    df_sim = pd.read_csv(file_log_sim_agg)
    return df_learn, df_sim


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
        path_to_experiments, "log", experiment_tag, "log_learn.csv"
    )
    df = pd.read_csv(file_log_learn)
    # get runtimes (min, max)
    df["time_total"] = df["time_init"] + df["time_run"]
    df = df.groupby(cols_index).agg(
        {"time_init": "first", "time_run": ["min", "max"], "time_total": ["min", "max"]}
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


def metric_to_str(mean: float, std: float, num_decimals: int = 3) -> str:
    return f"{mean:.{num_decimals}f} ({std:.{num_decimals}f})"


def create_str_col(
    df: pd.DataFrame, new_column: str, column: str, num_decimals: int = 3
) -> pd.DataFrame:
    df[new_column] = [
        metric_to_str(df[f"mean_{column}"][i], df[f"std_{column}"][i]) for i in df.index
    ]
    return df

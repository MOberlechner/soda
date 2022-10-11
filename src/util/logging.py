from datetime import datetime
from os.path import exists

import numpy as np
import pandas as pd

from src.util.metrics import (
    best_response_stability,
    monotonicity,
    next_iterate_stability,
    variational_stability,
)


def log_strat(
    strategies,
    cfg_learner,
    learn_alg: str,
    experiment: str,
    setting: str,
    run: int,
    time_init,
    time_run,
    path: str,
):
    """
    log results for each agent of the computation of strategies
    """

    # filename
    filename = "log.csv"

    # create new entry for each agent
    rows = [
        {
            "experiment": experiment,
            "mechanism": setting,
            "learner": learn_alg,
            "run": run,
            "agent": agent,
            "utility": strategies[agent].utility[-1],
            "utility_loss": strategies[agent].utility_loss[-1],
            "iterations": len(strategies[agent].utility),
            "convergence": strategies[agent].utility_loss[-1] < cfg_learner.tol,
            "iter/sec": round(len(strategies[agent].utility) / time_run, 2),
            "time_init": round(time_init, 2),
            "time_run": round(time_run, 2),
            "timestamp": str(datetime.now())[:-7],
        }
        for agent in strategies
    ]
    # create DataFrame with entries
    df = pd.DataFrame(rows)

    if exists(path + filename):
        # import existing dataframe
        log = pd.read_csv(path + filename)
        log = pd.concat([log, df])
    else:
        log = df

    # save data
    log.to_csv(path + filename, index=False)


def log_run(
    strategies,
    learn_alg: str,
    experiment: str,
    setting: str,
    run: int,
    path: str,
):
    """
    log variational stability and monotonicity for each run
    """

    # file name
    filename = "log_stability.csv"

    # create new entry for each agent
    rows = [
        {
            "experiment": experiment,
            "mechanism": setting,
            "learner": learn_alg,
            "run": run,
            "iterations": len(strategies[agent].utility),
            "var_stab_max": variational_stability(strategies).max(),
            "var_stab_bool": np.all(variational_stability(strategies) <= 0),
            "br_stab_max": best_response_stability(strategies).max(),
            "br_stab_bool": np.all(best_response_stability(strategies) <= 0),
            "ni_stab_max": next_iterate_stability(strategies).max(),
            "ni_stab_bool": np.all(next_iterate_stability(strategies) <= 0),
            "timestamp": str(datetime.now())[:-7],
        }
        for agent in strategies
    ]
    # create DataFrame with entries
    df = pd.DataFrame(rows)

    if exists(path + filename):
        # import existing dataframe
        log = pd.read_csv(path + filename)
        log = pd.concat([log, df])
    else:
        log = df

    # save data
    log.to_csv(path + filename, index=False)


def log_sim(
    strategies, experiment: str, setting: str, run: int, tag, values, path: str
):
    # filename
    filename = "log_sim.csv"

    # create new entry for each agent
    rows = [
        {
            "experiment": experiment,
            "mechanism": setting,
            "run": run,
            "agent": agent,
            "tag": tag,
            "value": values[agent],
            "timestamp": str(datetime.now())[:-7],
        }
        for agent in strategies
    ]
    # create DataFrame with entries
    df = pd.DataFrame(rows)

    if exists(path + filename):
        # import existing dataframe
        log = pd.read_csv(path + filename)
        log = pd.concat([log, df])
    else:
        log = df

    # save data
    log.to_csv(path + filename, index=False)


def agg_log_sim(path, decimal=4):
    """
    Aggregate results from log_sim: mean and std
    """

    # import log
    df = pd.read_csv(path + "log_sim.csv").drop(columns=["timestamp"])

    # aggregate over runs
    metrics = df["tag"].unique()
    df = pd.pivot(
        df,
        index=["experiment", "mechanism", "run", "agent"],
        columns="tag",
        values="value",
    ).reset_index()
    df1 = df.groupby(["experiment", "mechanism", "agent"]).agg(
        {tag: "mean" for tag in metrics}
    )
    df1.columns = [c + "_mean" for c in df1.columns]
    df2 = df.groupby(["experiment", "mechanism", "agent"]).agg(
        {tag: "std" for tag in metrics}
    )
    df2.columns = [c + "_std" for c in df1.columns]
    df = df1.merge(df2, how="outer", left_index=True, right_index=True)
    df = df.reindex(sorted(df.columns), axis=1).round(4)

    # save aggregate log file
    df.to_csv(path + "log_sim_agg.csv", index=False)

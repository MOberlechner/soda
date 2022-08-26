from datetime import datetime
from os.path import exists

import pandas as pd


def log_run(
    strategies,
    cfg_learner,
    experiment: str,
    setting: str,
    run: int,
    time_init,
    time_run,
    conv: bool,
    path: str,
):

    # create new entry for each agent
    rows = [
        {
            "experiment": experiment,
            "mechanism": setting,
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

    if exists(path + "log.csv"):
        # import existing dataframe
        log = pd.read_csv(path + "log.csv")
        log = pd.concat([log, df])
    else:
        log = df

    # save data
    log.to_csv(path + "log.csv", index=False)


def log_sim(
    strategies, experiment: str, setting: str, run: int, tag, values, path: str
):
    """ """

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

    if exists(path + "log_sim.csv"):
        # import existing dataframe
        log = pd.read_csv(path + "log_sim.csv")
        log = pd.concat([log, df])
    else:
        log = df

    # save data
    log.to_csv(path + "log_sim.csv", index=False)


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

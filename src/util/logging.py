from datetime import datetime
from os.path import exists

import pandas as pd


def log_run(
    strategies, experiment: str, setting: str, run: int, time_init, time_run, path: str
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


def log_sim(strategies, experiment: str, run: int, tag, values, path: str):

    # create new entry for each agent
    rows = [
        {
            "experiment": experiment,
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

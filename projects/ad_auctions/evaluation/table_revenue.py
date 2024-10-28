import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import pandas as pd

from projects.ad_auctions.config_exp import PATH_TO_EXPERIMENTS, PATH_TO_RESULTS


def create_table_sim(experiment_tag: str, round_dec: int) -> pd.DataFrame:

    # Parameter
    file_learn = os.path.join(
        PATH_TO_EXPERIMENTS, experiment_tag, "log", "computation_aggr.csv"
    )
    file_sim = os.path.join(
        PATH_TO_EXPERIMENTS, experiment_tag, "log", "simulation_aggr.csv"
    )
    index_cols = ["setting", "mechanism", "learner", "agent"]
    metric_cols_learn = ["utility_loss"]
    metric_cols_sim = ["util_loss", "l2_norm"]

    # Learning
    df1 = pd.read_csv(file_learn)
    df1 = df1[df1.metric.isin(metric_cols_learn)]
    df1 = pd.pivot(df1, index=index_cols, columns=["metric"], values=["mean", "std"])
    df1.columns = ["_".join(c) for c in df1.columns]
    df1 = df1.reset_index()

    # Simulation
    df2 = pd.read_csv(file_sim)
    df2 = df2[df2.metric.isin(metric_cols_sim)]
    df2 = pd.pivot(df2, index=index_cols, columns=["metric"], values=["mean", "std"])
    df2.columns = ["_".join(c) for c in df2.columns]
    df2 = df2.reset_index()

    # Create Table
    print(df1)
    print(df2)
    df = df1.merge(df2, how="outer", on=index_cols).reset_index()
    for c in metric_cols_learn + metric_cols_sim:
        for i in df.index:
            val_mean, val_std = df[f"mean_{c}"][i], df[f"std_{c}"][i]
            df.loc[i, c] = f"{val_mean:.3f} ({val_std:.3f})".replace("nan", "-")
    df["learner"] = [l.split("_")[0] for l in df.learner]
    df = df[index_cols + metric_cols_learn + metric_cols_sim]
    return df.sort_values(index_cols).reset_index(drop=True)


if __name__ == "__main__":

    LABEL_EXPERIMENT = "revenue"
    df = create_table_sim(LABEL_EXPERIMENT, round_dec=3)
    table = df.to_markdown(
        index=False, tablefmt="pipe", colalign=["center"] * len(df.columns)
    )
    print(f"\nTABLE REVENUE\n\n{table}\n")
    df.to_csv(
        os.path.join(PATH_TO_RESULTS, f"table_{LABEL_EXPERIMENT}.csv"), index=False
    )

    LABEL_EXPERIMENT = "revenue_asym"
    df = create_table_sim(LABEL_EXPERIMENT, round_dec=3)
    table = df.to_markdown(
        index=False, tablefmt="pipe", colalign=["center"] * len(df.columns)
    )
    print(f"\nTABLE REVENUE\n\n{table}\n")

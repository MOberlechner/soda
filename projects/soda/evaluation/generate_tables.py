import os

import numpy as np
import pandas as pd

from projects.soda.config_exp import (
    PATH_TO_EXPERIMENTS,
    PATH_TO_RESULTS,
    ROUND_DECIMALS_TABLE,
)
from soda.util.evaluation import create_table


def check_log_exists(experiment_tag: str) -> bool:
    files = ["computation_aggr.csv", "computation_aggr.csv", "simulation_aggr.csv"]
    return np.all(
        [
            os.path.exists(os.path.join(PATH_TO_EXPERIMENTS, experiment_tag, "log"))
            for f in files
        ]
    )


def save_table(df, table_nummer: int, label: str = "") -> None:
    table = df.to_markdown(
        index=False, tablefmt="pipe", colalign=["center"] * len(df.columns)
    )
    path_table = os.path.join(
        os.path.join(PATH_TO_RESULTS, "tables", f"table_{table_nummer:02.0f}.txt")
    )
    f = open(path_table, "w")
    f.write(table)
    f.close()


def generate_table_interdependent():
    """Tables for common and affiliated values model (table 3 & 4)"""
    if check_log_exists("interdependent"):
        df = create_table(
            PATH_TO_EXPERIMENTS, "interdependent", num_decimals=ROUND_DECIMALS_TABLE
        )
        df_com = df[df.setting == "common_value"].drop(columns=["util_loss_discr"])
        df_aff = df[df.setting == "affiliated_values"].drop(columns=["util_loss_discr"])
        save_table(df_com, 3)
        save_table(df_aff, 4)


def generate_table_llg():
    """Tables for LLG auction (table 5, 6, 7)"""
    if check_log_exists("llg"):
        df = create_table(PATH_TO_EXPERIMENTS, "llg", num_decimals=ROUND_DECIMALS_TABLE)
        for i, pr in enumerate(["nz", "nvcg", "nb"]):
            table = df[
                df.setting.isin([f"{pr}_gamma{i}" for i in [1, 2, 3]])
                & (df.agent == "L")
            ].drop(columns=["util_loss_discr"])
            save_table(table, 5 + i)


def generate_table_split_award():
    """Tables for Split Award auction (table 8, 9)"""
    if check_log_exists("split_award"):
        df = create_table(
            PATH_TO_EXPERIMENTS, "split_award", num_decimals=ROUND_DECIMALS_TABLE
        )
        table = df[df.setting == "sa_gaussian"].drop(columns=["util_loss_discr"])
        save_table(table, 8)
        table = df[df.setting == "sa_uniform"].drop(columns=["util_loss_discr"])
        save_table(table, 9)


def generate_table_risk():
    """Results for FPSB with risk aversion (table 10)"""
    if check_log_exists("risk"):
        settings = ["fpsb_risk0", "fpsb_risk2", "fpsb_risk4"]
        df = create_table(
            PATH_TO_EXPERIMENTS, "risk", num_decimals=ROUND_DECIMALS_TABLE
        )
        df = df[df.setting.isin(settings)].drop(columns=["util_loss_discr"])
        save_table(df, 10)
    else:
        print("Log-files for experiments 'risk' not found")


def generate_table_discretization():
    """Results for FPSB with different levels of discretization (table 11)"""
    if check_log_exists("discretization"):
        discr = ["016", "032", "064", "128", "256"]
        settings = [f"fast_2_discr{d}" for d in discr]
        df = create_table(
            PATH_TO_EXPERIMENTS, "discretization", num_decimals=ROUND_DECIMALS_TABLE
        )
        df = df[df.setting.isin(settings)].drop(columns=["util_loss_discr"])
        save_table(df, 11)
    else:
        print("Log-files for experiments 'discretization' not found")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS, "tables"), exist_ok=True)

    generate_table_interdependent()
    generate_table_llg()
    generate_table_split_award()
    generate_table_risk()
    generate_table_discretization()

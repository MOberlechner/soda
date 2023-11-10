import os

import numpy as np
import pandas as pd

from projects.soda.config_exp import PATH_TO_EXPERIMENTS, PATH_TO_RESULTS
from soda.util.evaluation import create_table


def check_log_exists(experiment_tag: str) -> bool:
    files = ["log_learn_agg.csv", "log_learn.csv", "log_sim_agg.csv"]
    return np.all(
        [
            os.path.exists(os.path.join(PATH_TO_EXPERIMENTS, "log", experiment_tag))
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
        df = create_table(PATH_TO_EXPERIMENTS, "interdependent")
        df_com = df[df.setting == "common_value"].drop(columns=["util_loss_discr"])
        df_aff = df[df.setting == "affiliated_values"].drop(columns=["util_loss_discr"])
        save_table(df_com, 3)
        save_table(df_aff, 4)


def generate_table_llg():
    """Tables for LLG auction (table 5, 6, 7)"""
    if check_log_exists("llg"):
        df = create_table(PATH_TO_EXPERIMENTS, "llg")
        for i, pr in enumerate(["nb", "nvcg", "nz"]):
            table = df[df.setting == f"llg_auction_{pr}"]
            save_table(table, 5 + i)


def generate_table_split_award():
    """Tables for Split Award auction (table 8, 9)"""
    pass


def generate_table_risk():
    """Results for FPSB with risk aversion (table 10)"""
    if check_log_exists("risk"):
        settings = ["fpsb_risk0", "fpsb_risk2", "fpsb_risk4"]
        df = create_table(PATH_TO_EXPERIMENTS, "risk")
        df = df[df.setting.isin(settings)].drop(columns=["util_loss_discr"])
        save_table(df, 10)
    else:
        print("Log-files for experiments 'risk' not found")


if __name__ == "__main__":
    os.makedirs(os.path.join(PATH_TO_RESULTS, "tables"), exist_ok=True)

    generate_table_interdependent()
    generate_table_llg()
    generate_table_split_award()
    generate_table_risk()

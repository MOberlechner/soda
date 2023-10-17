from datetime import datetime
from os.path import exists
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from src.strategy import Strategy


class Logger:
    """
    Logger collects information on experiments
    """

    def __init__(
        self,
        path: str,
        setting: str,
        experiment: str,
        learn_alg: str,
        logging: bool,
        round_decimal: int = 4,
    ):
        """Initialize Logger

        Args:
            path (str): path to csv file which gets created
            setting (str): mechanism/setting for experiment
            experiment (str): experiment
            learn_alg (str): used learning algorithm
            logging (bool): store results in csv
        """
        self.path = path + "log/" + setting + "/"
        self.setting = setting
        self.experiment = experiment
        self.learn_alg = learn_alg
        self.logging = logging
        self.round_decimal = round_decimal

        # create directory for each setting
        Path(self.path).mkdir(parents=True, exist_ok=True)

        # log run learning
        self.filename_log_learning = "log_learn.csv"
        self.filename_log_learning_agg = "log_learn_agg.csv"
        self.file_log_learning = pd.DataFrame(
            columns=[
                "experiment",
                "mechanism",
                "learner",
                "run",
                "agent",
                "utility",
                "utility_loss",
                "dist_prev_iter",
                "iterations",
                "convergence",
                "iter/sec",
                "time_init",
                "time_run",
                "timestamp",
            ]
        )

        # log run simulation
        self.filename_log_simulation = "log_sim.csv"
        self.filename_log_simulation_agg = "log_sim_agg.csv"
        self.file_log_simulation = pd.DataFrame(
            columns=[
                "experiment",
                "mechanism",
                "learner",
                "run",
                "agent",
                "tag",
                "value",
                "timestamp",
            ]
        )

    # -------------------- methods to log results from computation --------------------

    def log_learning(self) -> None:
        """Main function to save logs from learning experiment.
        Should be called for each experiment after all runs.
        """
        if self.logging:
            self.save_learning_run()
            self.save_learning_aggr()
        else:
            print("Results not logged.")

    def save_learning_run(self) -> None:
        """Save csv from learning experiment"""
        if exists(self.path + self.filename_log_learning):
            log = pd.read_csv(self.path + self.filename_log_learning)
            log = pd.concat([log, self.file_log_learning])
        else:
            log = self.file_log_learning
        log.to_csv(self.path + self.filename_log_learning, index=False)

    def log_learning_run(
        self,
        strategies: Dict[str, Strategy],
        run: int,
        convergence: bool,
        iteration: int,
        time_init: float,
        time_run: float,
    ) -> None:
        """Log metrics created by learning process

        Args:
            strategies: computed strategy
            run (int): run, i.e., repetition of experiment
            convergence (bool): did method converge
            iteration (int):  number of iterations
            time_init (float): time to setup game (includes utility computation)
            time_run (float): time to run learning algorithm
        """
        # entries
        rows = [
            {
                "experiment": self.experiment,
                "mechanism": self.setting,
                "learner": self.learn_alg,
                "run": run,
                "agent": agent,
                "utility": strategies[agent].utility[iteration],
                "utility_loss": strategies[agent].utility_loss[iteration],
                "dist_prev_iter": strategies[agent].dist_prev_iter[iteration],
                "iterations": iteration,
                "convergence": float(convergence),
                "iter/sec": round(len(strategies[agent].utility) / time_run, 2),
                "time_init": round(time_init, 2),
                "time_run": round(time_run, 2),
                "timestamp": str(datetime.now())[:-7],
            }
            for agent in strategies
        ]
        df_rows = pd.DataFrame(rows)
        # add entry to DataFrame
        self.file_log_learning = pd.concat([self.file_log_learning, df_rows])

    def save_learning_aggr(self) -> None:
        """Save aggregated csv from learning experiment"""
        df = self.log_learning_aggr()
        if df is not None:
            if exists(self.path + self.filename_log_learning_agg):
                log = pd.read_csv(self.path + self.filename_log_learning_agg)
                log = pd.concat([log, df])
            else:
                log = df
            log.to_csv(self.path + self.filename_log_learning_agg, index=False)

    def log_learning_aggr(self):
        """Aggregate file_log_learning if there are more runs"""
        if self.file_log_learning["run"].max() > 0:
            indices = ["experiment", "mechanism", "learner", "agent"]
            columns = [
                "convergence",
                "iterations",
                "utility",
                "utility_loss",
                "dist_prev_iter",
                "iter/sec",
                "time_run",
                "time_init",
            ]
            # mean
            df_mean = (
                self.file_log_learning.groupby(indices)
                .agg({c: "mean" for c in columns})
                .reset_index()
            )
            df_mean = df_mean.melt(indices, var_name="Metric", value_name="Mean")
            # std
            df_std = (
                self.file_log_learning.groupby(indices)
                .agg({c: "std" for c in columns[:-1]})
                .reset_index()
            )
            df_std = df_std.melt(indices, var_name="Metric", value_name="Std")
            # combine
            return (
                df_mean.merge(df_std, on=indices + ["Metric"], how="outer")
                .sort_values(indices + ["Metric"])
                .round(self.round_decimal)
            )
        else:
            return None

    # -------------------- methods to log results from simulation --------------------

    def log_simulation(self) -> None:
        """Main function to save logs from simulation experiment.
        Should be called for each experiment after all runs.
        """
        if self.logging:
            self.save_simulation_run()
            self.save_simulatioa_aggr()
        else:
            print("Results not logged.")

    def save_simulation_run(self) -> None:
        """Save csv from learning experiment"""
        if exists(self.path + self.filename_log_simulation):
            log = pd.read_csv(self.path + self.filename_log_simulation)
            log = pd.concat([log, self.file_log_simulation])
        else:
            log = self.file_log_simulation
        log.to_csv(self.path + self.filename_log_simulation, index=False)

    def log_simulation_run(self, run: int, agent: str, tag: str, value: dict):
        """Log metric created by simulation process

        Args:
            run (int): run, i.e., repetition of experiment
            tag (str): specifies metric
            values (dict): values for all agents
        """
        # entries
        row = [
            {
                "experiment": self.experiment,
                "mechanism": self.setting,
                "learner": self.learn_alg,
                "run": run,
                "agent": agent,
                "tag": tag,
                "value": value,
                "timestamp": str(datetime.now())[:-7],
            }
        ]
        # add entrie to DataFrame
        self.file_log_simulation = pd.concat(
            [self.file_log_simulation, pd.DataFrame(row)]
        )

    def save_simulatioa_aggr(self) -> None:
        """Save aggregated csv from simulation experiment"""
        df = self.log_simulation_aggr()
        if df is not None:
            if exists(self.path + self.filename_log_simulation_agg):
                log = pd.read_csv(self.path + self.filename_log_simulation_agg)
                log = pd.concat([log, df])
            else:
                log = df
            log.to_csv(self.path + self.filename_log_simulation_agg, index=False)

    def log_simulation_aggr(self) -> pd.DataFrame:
        if self.file_log_simulation["run"].max() > 0:
            indices = ["experiment", "mechanism", "learner", "agent", "tag"]
            # mean
            df_mean = (
                self.file_log_simulation.groupby(indices)
                .agg({"value": "mean"})
                .reset_index()
            )
            df_mean = df_mean.rename(columns={"tag": "metric", "value": "mean"})
            # std
            df_std = (
                self.file_log_simulation.groupby(indices)
                .agg({"value": "std"})
                .reset_index()
            )
            df_std = df_std.rename(columns={"tag": "metric", "value": "std"})
            # combine
            return (
                df_mean.merge(
                    df_std,
                    on=["experiment", "mechanism", "learner", "agent", "metric"],
                    how="outer",
                )
                .sort_values(["experiment", "mechanism", "learner", "agent", "metric"])
                .round(self.round_decimal)
            )
        else:
            return None

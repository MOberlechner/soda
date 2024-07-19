import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from soda.strategy import Strategy


class Logger:
    """
    Logger collects information on experiments
    """

    def __init__(
        self,
        param_logging: dict,
        mechanism: str,
        setting: str,
        learn_alg: str,
    ):
        """Initialize Logger

        Args:
            path_log (str): path to csv file which gets created
            experiment_tag (str): setting can denote a group of experiments. If None, we group experiments by mechanism
            mechanism (str): mechanism for experiment
            setting (str): setting (specific instance of mechanism)
            learn_alg (str): used learning algorithm
            logging (bool): store results in csv
        """
        #  we group experiments by setting or mechanism if setting is not defined
        self.path = path_log
        self.experiment_tag = experiment_tag
        self.mechanism = mechanism
        self.setting = setting
        self.learn_alg = learn_alg
        self.logging = logging
        self.round_decimal = round_decimal

        # log run learning
        self.filename_log_learning = os.path.join(self.path, "log_learn.csv")
        self.filename_log_learning_agg = os.path.join(self.path, "log_learn_agg.csv")
        self.file_log_learning = pd.DataFrame(
            columns=[
                "setting",
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
        self.filename_log_simulation = os.path.join(self.path, "log_sim.csv")
        self.filename_log_simulation_agg = os.path.join(self.path, "log_sim_agg.csv")
        self.file_log_simulation = pd.DataFrame(
            columns=[
                "setting",
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
        if os.path.exists(self.filename_log_learning):
            log = pd.read_csv(self.filename_log_learning)
            log = pd.concat([log, self.file_log_learning])
        else:
            log = self.file_log_learning
        log.to_csv(self.filename_log_learning, index=False)

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
                "setting": self.setting,
                "mechanism": self.mechanism,
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
            if os.path.exists(self.filename_log_learning_agg):
                log = pd.read_csv(self.filename_log_learning_agg)
                log = pd.concat([log, df])
            else:
                log = df
            log.to_csv(self.filename_log_learning_agg, index=False)

    def log_learning_aggr(self):
        """Aggregate file_log_learning if there are more runs"""
        if self.file_log_learning["run"].max() > 0:
            indices = ["setting", "mechanism", "learner", "agent"]
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
            df_mean = df_mean.melt(indices, var_name="metric", value_name="mean")
            # std
            df_std = (
                self.file_log_learning.groupby(indices)
                .agg({c: "std" for c in columns[:-1]})
                .reset_index()
            )
            df_std = df_std.melt(indices, var_name="metric", value_name="std")
            # combine
            return (
                df_mean.merge(df_std, on=indices + ["metric"], how="outer")
                .sort_values(indices + ["metric"])
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
            self.save_simulation_aggr()
        else:
            print("Results not logged.")

    def save_simulation_run(self) -> None:
        """Save csv from learning experiment"""
        if os.path.exists(self.filename_log_simulation):
            log = pd.read_csv(self.filename_log_simulation)
            log = pd.concat([log, self.file_log_simulation])
        else:
            log = self.file_log_simulation
        log.to_csv(self.filename_log_simulation, index=False)

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
                "setting": self.setting,
                "mechanism": self.mechanism,
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

    def save_simulation_aggr(self) -> None:
        """Save aggregated csv from simulation experiment"""
        df = self.log_simulation_aggr()
        if df is not None:
            if os.path.exists(self.filename_log_simulation_agg):
                log = pd.read_csv(self.filename_log_simulation_agg)
                log = pd.concat([log, df])
            else:
                log = df
            log.to_csv(self.filename_log_simulation_agg, index=False)

    def log_simulation_aggr(self) -> pd.DataFrame:
        if self.file_log_simulation["run"].max() > 0:
            indices = ["setting", "mechanism", "learner", "agent", "tag"]
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
                    on=["setting", "mechanism", "learner", "agent", "metric"],
                    how="outer",
                )
                .sort_values(["setting", "mechanism", "learner", "agent", "metric"])
                .round(self.round_decimal)
            )
        else:
            return None

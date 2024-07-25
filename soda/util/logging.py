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
        label_mechanism: str,
        label_setting: str,
        label_learner: str,
        label_experiment: str,
        param_logging: dict,
    ):
        """Initialize Logger

        Args:
            label_mechanism (str): mechanism for experiment
            label_setting (str): setting (specific instance of mechanism)
            label_learner (str): used learning algorithm
            label_experiment (str): used to structure experiments
            param_logging (dict): parameter for logging
                - save_strategy
                - save_strategy_init
        """

        self.label_mechanism = label_mechanism
        self.label_setting = label_setting
        self.learn_learner = learn_learner

        # create directories
        self.path = os.path.join(param_logging["path_experiment"], label_experiment)

        # create empty dataframes for logs
        self.data = self.init_data()

    # -------------------- method to save logs from experiment --------------
    def init_data(self) -> None:
        """create data frames to save logs"""
        self.columns = {
            "default": ["setting", "mechanism", "learner", "run", "agent", "timestamp"],
            "learn": [
                "utility",
                "utility_loss",
                "dist_prev_iter",
                "iterations",
                "convergence",
                "iter/sec",
                "time_init",
                "time_run",
            ],
            "sim": ["tag", "value"],
            "eval": ["utility", "utility_loss"],
        }
        return {
            sub_exp: pd.DataFrame(columns=self.columns["default"] + columns[sub_exp])
            for sub_exp in ["learn", "sim", "eval"]
        }

    def get_log_path(self, sub_exp: str, aggregate: bool = False) -> Path:
        """Path to log files for different subexperiments"""
        assert sub_exp in ["learn", "sim", "eval"]
        file_name = f"log_{sub_exp}" + ("_agg" if aggregate else "")
        return os.path.join(self.path, f"{file_name}.csv")

    def get_log_file(self, sub_exp: str, aggregate: bool = False) -> pd.DataFrame:
        """Get data frame with logs for different subexperiments"""
        assert sub_exp in ["learn", "sim", "eval"]
        log_name = f"log_{sub_exp}" + ("_agg" if aggregate else "")
        return self.data[log_name]

    def save_sub_experiment(self, sub_exp) -> None:
        """Save logging to csv file (append if file already exists)"""
        for aggregate in [False, True]:
            path_file = self.get_log_path(sub_exp, aggregate)
            log_file = self.get_log_file(sub_exp, aggregate)
            if os.path.exists(path_file):
                log = pd.read_csv(path_file)
                log = pd.concat([log, log_file])
            else:
                log = log_file
            log.to_csv(path_file, index=False)

    # -------------------- methods to log results from computation --------------------

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
                "setting": self.label_settingsetting,
                "mechanism": self.label_mechanismmechanism,
                "learner": self.label_learner,
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

    def save_strategies(self, strategies: Dict[Strategy], run: int) -> None:
        """save computed strategies from computation result"""
        name = f"{self.label_learner}_{self.label_setting}_run_{run}"
        if self.param_logging["save_strategy"]:
            for i in self.strategies:
                self.strategies[i].save(
                    name=name,
                    path=self.path_strat,
                    save_init=self.param_logging["save_strategy_init"],
                )

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

    # -------------------- methods to log results from evaluation --------------------

    def log_evaluation(self) -> None:
        """Main function to save logs from evaluation experiment.
        Should be called for each experiment after all runs.
        """
        if self.logging:
            self.save_evaluation_run()
            self.save_evaluation_aggr()
        else:
            print("Results not logged.")

    def save_evaluation_run(self) -> None:
        """Save csv from learning experiment"""
        if os.path.exists(self.filename_log_simulation):
            log = pd.read_csv(self.filename_log_simulation)
            log = pd.concat([log, self.file_log_simulation])
        else:
            log = self.file_log_simulation
        log.to_csv(self.filename_log_simulation, index=False)

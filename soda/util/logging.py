import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from soda.learner.learner import Learner
from soda.strategy import Strategy


class Logger:
    """
    For each experiment, we log data per run for computation, simulation, and evaluation.
    After all runs, the data is saved in the respective directory.
    In general, we save the data per run, but also an aggregated version over all runs.

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
                - path_experiment (str)
                - save_strategy (bool)
                - save_strategy_init (bool)
                - save_images (bool)
                - round_decimal (int)

        Methods (which could be called from experiments.py)
            log_computation
            log_simulation
            log_evaluation
            save
            get_filename_strategy
        """

        self.label_mechanism = label_mechanism
        self.label_setting = label_setting
        self.label_learner = label_learner
        self.label_experiment = label_experiment

        self.check_input(param_logging)
        self.path_experiment = param_logging["path_experiment"]
        self.save_strategy = param_logging["save_strategy"]
        self.save_strategy_init = param_logging["save_strategy_init"]
        self.save_image = param_logging["save_image"]
        self.round_decimal = param_logging["round_decimal"]
        self.sub_experiments = ["computation", "simulation", "evaluation"]

        # create directories and initialize dataframe to store information
        self.create_directories()
        self.data = self.init_data()

    # -------------------- create and save dataframes with logging data --------------

    def save(self, sub_exp: str):
        """
        Logs are saved into two files
            - logs with data per run (e.g., computation.csv)
            - logs with aggregated data over all runs (e.g., computation_aggr.csv)
        If files already exist, new entries are appended.
        """
        assert sub_exp in self.sub_experiments
        self.aggregate_runs(sub_exp)
        self.save_sub_experiment(sub_exp)

    def save_sub_experiment(self, sub_exp) -> None:
        """Save logging to csv file (append if file already exists)"""
        for aggregate in [False, True]:
            path_file = self.get_log_filename(sub_exp, aggregate)
            log_file = self.get_log_file(sub_exp, aggregate)
            if os.path.exists(path_file):
                log = pd.read_csv(path_file)
                log = pd.concat([log, log_file])
            else:
                log = log_file
            log.round(self.round_decimal).to_csv(path_file, index=False)

    def get_log_filename(self, sub_exp: str, aggregate: bool = False) -> Path:
        """filename (with path) for log files (csv)"""
        file_name = f"{sub_exp}" + ("_aggr" if aggregate else "")
        return os.path.join(self.get_path("log"), f"{file_name}.csv")

    def get_log_file(self, sub_exp: str, aggregate: bool = False) -> pd.DataFrame:
        """import existing log files (csv)"""
        log_name = f"{sub_exp}" + ("_aggr" if aggregate else "")
        return self.data[log_name]

    def init_data(self) -> None:
        """create data frames to save logs"""
        return {sub_exp: pd.DataFrame() for sub_exp in self.sub_experiments}

    # -------------------- log data for subexperiments--------------------

    def log_run(
        self,
        sub_exp: str,
        run: int,
        agent: str,
        data: Dict[str, float],
    ):
        """general logging method that writes data into respective data frame"""
        if data:
            # get all logging data in dict
            entries = self.get_standard_entries(agent, run)
            entries.update(data)
            # write data in corresponding data frame
            df_row = pd.DataFrame([entries])
            self.data[sub_exp] = pd.concat([self.data[sub_exp], df_row])

    def get_standard_entries(self, agent, run) -> Dict:
        """get standard entries for general logging method log_run()"""
        return {
            "setting": self.label_setting,
            "mechanism": self.label_mechanism,
            "learner": self.label_learner,
            "run": run,
            "agent": agent,
            "timestamp": str(datetime.now())[:-7],
        }

    def log_computation(
        self,
        strategies: Dict[str, Strategy],
        learner: Learner,
        run: int,
        time_init: float,
        time_run: float,
    ) -> None:
        """Log metrics generated by subexperiment COMPUTATION

        Args:
            strategies: computed strategy
            learner: learner used to compute strategy
            run (int): run, i.e., repetition of experiment
            time_init (float): time to setup game (includes utility computation)
            time_run (float): time to run learning algorithm
        """
        for agent in strategies:
            data = {
                "utility": strategies[agent].utility[learner.iter],
                "utility_loss": strategies[agent].utility_loss[learner.iter],
                "dist_prev_iter": strategies[agent].dist_prev_iter[learner.iter],
                "iterations": learner.iter + 1,
                "convergence": float(learner.convergence),
                "iter/sec": round(learner.iter / time_run, 2),
                "time_init": round(time_init, 2),
                "time_run": round(time_run, 2),
            }
            self.log_run(sub_exp="computation", agent=agent, run=run, data=data)

            # save strategies and images
            self.save_strategies(strategies[agent], run)
            self.save_images(strategies[agent], run)

    def log_simulation(self, run: int, agent: str, data: Dict[str, float]):
        """Log metrics generated by subexperiment SIMULATION

        Args:
            run (int): run, i.e., repetition of experiment
            agent (str): number of agent
            data (List[str]): dict with metrics (str) and corresponding values (float)
        """
        self.log_run(sub_exp="simulation", run=run, agent=agent, data=data)

    def log_evaluation(self, run: int, agent: str, data: Dict[str, float]):
        """Log metrics generated by subexpeirment EVALUATION

        Args:
            run (int): run, i.e., repetition of experiment
            agent (str): number of agent
            data (List[str]): dict with metrics (str) and corresponding values (float)
        """
        self.log_run(sub_exp="simulation", run=run, agent=agent, data=data)

    def aggregate_runs(self, sub_exp: str) -> None:
        """aggregate log data over runs"""
        # compute aggregated log files
        index_columns = ["setting", "mechanism", "learner", "agent"]
        data_aggr = {}
        for statistic in ["mean", "std"]:
            data_aggr[statistic] = (
                self.data[sub_exp]
                .drop(columns="run")
                .groupby(index_columns)
                .agg(statistic)
                .reset_index()
            ).melt(index_columns, var_name="metric", value_name=statistic)

        # store aggregated log files in self.data
        self.data[f"{sub_exp}_aggr"] = (
            data_aggr["mean"]
            .merge(data_aggr["std"], on=index_columns + ["metric"], how="outer")
            .dropna(subset="mean")
            .sort_values(index_columns + ["metric"])
        )

    # -------------------- methods to handle strategies --------------------

    def save_strategies(self, strategy: Strategy, run: int) -> None:
        """save strategy
        example filename: setting_learner_run_0_agent_1.npy
        """
        if self.save_strategy:
            name = self.get_filename_strategy(run)
            filename = os.path.join(self.get_path("strategies"), name)
            strategy.save(filename=filename, save_init=self.save_strategy_init)

    def load_strategies(self, strategy: Strategy, run: int) -> Strategy:
        """load strategy"""
        name = self.get_filename_strategy(run)
        filename = os.path.join(self.get_path("strategies"), name)
        strategy.load(filename=filename)

    def save_images(self, strategy: Strategy, run) -> None:
        """save image of strategy
        filename identical to save_strategies()
        """
        if self.save_image:
            name = self.get_filename_strategy(run)
            filename = os.path.join(self.get_path("images"), name)
            strategy.plot(metrics=True, grad=True, save=True, save_path=filename)

    def get_filename_strategy(self, run: int) -> str:
        return f"{self.label_setting}_{self.label_learner}_run_{run}"

    # -------------------- helper methods ---------------------

    def check_input(self, param_logging) -> None:
        keys = [
            "path_experiment",
            "save_strategy",
            "save_strategy_init",
            "save_image",
            "round_decimal",
        ]
        for key in keys:
            if key not in param_logging:
                raise ValueError(f"key {key} is missing in param_logging")

    def get_path(self, subdir: str) -> Path:
        assert subdir in ["log", "strategies", "images"]
        return os.path.join(self.path_experiment, self.label_experiment, subdir)

    def create_directories(self) -> None:
        """create all necessary subdirectories"""
        Path(self.get_path("log")).mkdir(parents=True, exist_ok=True)
        if self.save_strategy:
            Path(self.get_path("strategies")).mkdir(parents=True, exist_ok=True)
        if self.save_image:
            Path(self.get_path("images")).mkdir(parents=True, exist_ok=True)

import os
from pathlib import Path
from time import time

import numpy as np
from src.util.config import Config
from src.util.logging import Logger
from tqdm import tqdm


class Experiment:
    """
    Config class to run experiments

    """

    def __init__(
        self,
        config_game: str,
        config_learner: str,
        number_runs: int,
        learning: bool,
        simulation: bool,
        logging: bool,
        save_strat: bool,
        number_samples: int,
        save_init_strat: bool,
        path_exp: str,
        round_decimal: int,
    ) -> None:
        """Initialize Experiment

        Args:
            config_game (str): config file for game/mechanism
            config_learner (str): config file for learner
            number_runs (int): number of repetitions of experiment
            learning (bool): run learning to learn strategies
            simulation (bool): run simulation to get metrics
            logging (bool): log results
            save_strat (bool): save strategies
            save_init_strat (bool, optional): path to directory containing the config files. Defaults to True.
            path_exp (str, optional): path to directory to save results. Defaults to "".
            round_decimal (int, optional): accuracy of metric in logger. Defaults to 3.
        """

        self.config_game = config_game
        self.config_learner = config_learner

        self.number_runs = number_runs
        self.learning = learning
        self.simulation = simulation
        self.number_samples = number_samples

        self.save_strat = save_strat
        self.save_init_strat = save_init_strat
        self.logging = logging
        self.path_exp = self._get_path(path_exp)

        print(f"Experiment started".ljust(100, "."))
        print(f" - game:    {self.config_game}\n - learner: {self.config_learner}")

        try:
            # setup game and learner using config
            self.config = Config(self.config_game, self.config_learner)
            self.game, self.learner = self.config.create_setting()

            # initialize logger
            self.label_mechanism = self.game.name
            self.label_experiment = Path(self.config_game).stem
            self.label_learner = Path(self.config_learner).stem
            self.logger = Logger(
                self.path_exp,
                self.label_mechanism,
                self.label_experiment,
                self.label_learner,
                logging,
                round_decimal,
            )

            # create directories (if necessary)
            if self.logging or self.save_strat:
                if self.path_exp == "":
                    raise ValueError("path to store strategies/log-files not given")
                Path(self.path_exp + "strategies/" + self.label_mechanism).mkdir(
                    parents=True, exist_ok=True
                )
            print(f" - Setting created ")

        except Exception as e:
            print(e)
            print(f" - Error: setting not created")
            self.learning, self.simulation = False, False

    def run(self) -> None:
        """run experiment, i.e., learning and simulation"""
        # run learning
        if self.learning:
            try:
                self.run_learning()
            except Exception as e:
                print(e)
                print(" - Error in Learning ")

        # run simulation
        if self.simulation:
            try:
                self.run_simulation()
            except Exception as e:
                print(e)
                print("- Error in Simulation")
        print(f"Done ".ljust(100, ".") + "\n")

    def run_learning(self) -> None:
        """run learning of strategies"""
        print(" - Learning:")
        t0 = time()
        if not self.game.mechanism.own_gradient:
            self.game.get_utility()
            print(" - utilities of approximation game computed")
        time_init = time() - t0

        # repeat experiment
        for run in tqdm(
            range(self.number_runs),
            unit_scale=True,
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}",
            colour="green",
            desc="   Progress",
        ):

            # init strategies
            self.strategies = self.config.create_strategies(self.game)

            # run learning algorithm
            t0 = time()
            self.learner.run(self.game, self.strategies)
            time_run = time() - t0

            # log and save
            self.logger.log_learning_run(
                self.strategies,
                run,
                self.learner.convergence,
                self.learner.iter,
                time_init,
                time_run,
            )
            self.save_strategies(run)

        self.logger.log_learning()

    def run_simulation(self) -> None:
        """run simulation for computed strategies"""
        print(" - Simulation:")

        # repeat experiment
        for run in tqdm(
            range(self.number_runs),
            unit_scale=True,
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}",
            colour="blue",
            desc="   Progress",
        ):

            # load computed strategies
            self.load_strategies(run)

            if all(self.strategies[i].x is not None for i in self.game.set_bidder):

                # sample observation and bids
                obs_profile = self.game.mechanism.sample_types(self.number_samples)
                bid_profile = np.array(
                    [
                        self.strategies[self.game.bidder[i]].sample_bids(obs_profile[i])
                        for i in range(self.game.n_bidder)
                    ]
                )

                # compute metrics
                for i in self.game.set_bidder:
                    metrics, values = self.game.mechanism.get_metrics(
                        i, obs_profile, bid_profile
                    )
                    for tag, val in zip(metrics, values):
                        self.logger.log_simulation_run(run, i, tag, val)
            else:
                break

        self.logger.log_simulation()
        print(" - run_simulation finished")

    def save_strategies(self, run: int) -> None:
        """Save strategies for current experiment

        Args:
            run (int): current repetition of experiment
        """
        if self.save_strat:
            name = f"{self.label_learner}_{self.label_experiment}_run_{run}"
            for i in self.strategies:
                self.strategies[i].save(
                    name=name,
                    setting=self.label_mechanism,
                    path=self.path_exp,
                    save_init=self.save_init_strat,
                )

    def load_strategies(self, run: int) -> None:
        """Load strategies for current experiment

        Args:
            run (int): current repetition of experiment
        """
        # init strategies
        self.strategies = self.config.create_strategies(self.game)
        name = f"{self.label_learner}_{self.label_experiment}_run_{run}"
        for i in self.strategies:
            self.strategies[i].load(
                name=name,
                setting=self.label_mechanism,
                path=self.path_exp,
            )

    def _get_path(self, path_dir: str = "experiments/test/") -> str:
        """Get path to project (soda) and directory for experiment

        Args:
            path_dir (str): path from project directory to experiment directory

        Returns:
            path (str): path to experiment directory
        """
        path_into_project = os.getcwd().split("soda")[0] + "soda/"
        return path_into_project + path_dir

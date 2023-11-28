import os
from pathlib import Path
from time import time

import numpy as np
from tqdm import tqdm

from soda.util.config import Config
from soda.util.logging import Logger


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
        path_exp: str,
        round_decimal: int = 5,
        experiment_tag: str = "",
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
            path_exp (str, optional): path to directory to save results.
            round_decimal (int, optional): accuracy of metric in logger. Defaults to 5.
            experiment_tag (str, optional): name for a group of different settings. Allows us to structure our results in different experiments. If not specified, we group the results by mechanism.
        """

        self.config_game = config_game
        self.config_learner = config_learner

        self.number_runs = number_runs
        self.learning = learning
        self.simulation = simulation
        self.number_samples = number_samples

        self.save_strat = save_strat
        self.logging = logging

        print(f"Experiment started".ljust(100, "."))
        print(f" - game:    {self.config_game}\n - learner: {self.config_learner}")

        try:
            # setup game and learner using config
            self.config = Config(self.config_game, self.config_learner)
            self.game, self.learner = self.config.create_setting()

            self.label_mechanism = self.game.name
            self.experiment_tag = (
                experiment_tag if experiment_tag != "" else self.label_mechanism
            )
            self.label_setting = Path(self.config_game).stem
            self.label_learner = Path(self.config_learner).stem

            # create directories (if necessary)
            self.path_log = os.path.join(path_exp, "log", self.experiment_tag)
            self.path_strat = os.path.join(path_exp, "strategies", self.experiment_tag)

            if self.logging:
                Path(self.path_log).mkdir(parents=True, exist_ok=True)

            if self.save_strat:
                Path(self.path_strat).mkdir(parents=True, exist_ok=True)

            # initialize logger
            self.logger = Logger(
                self.path_log,
                self.experiment_tag,
                self.label_mechanism,
                self.label_setting,
                self.label_learner,
                logging,
                round_decimal,
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
            print("    utilities of approximation game computed")
        time_init = time() - t0

        # repeat experiment
        for run in tqdm(
            range(self.number_runs),
            unit_scale=True,
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}",
            colour="green",
            desc="    Progress",
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
            desc="    Progress",
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
            name = f"{self.label_learner}_{self.label_setting}_run_{run}"
            for i in self.strategies:
                self.strategies[i].save(
                    name=name,
                    path=self.path_strat,
                    save_init=True,
                )

    def load_strategies(self, run: int) -> None:
        """Load strategies for current experiment

        Args:
            run (int): current repetition of experiment
        """
        # init strategies
        self.strategies = self.config.create_strategies(self.game)
        name = f"{self.label_learner}_{self.label_setting}_run_{run}"
        for i in self.strategies:
            self.strategies[i].load(
                name=name,
                path=self.path_strat,
            )

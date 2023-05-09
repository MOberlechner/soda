from pathlib import Path
from time import time

import numpy as np
from tqdm import tqdm

from src.util.config import Config
from src.util.logging import Logger


class Experiment:
    """
    Config class to run experiments

    """

    def __init__(
        self,
        mechanism_type: str,
        experiment: str,
        learn_alg: str,
        number_runs: int,
        learning: bool,
        simulation: bool,
        logging: bool,
        save_strat: bool,
        path_config: str,
        number_samples: int,
        save_init_strat: bool,
        path_exp: str,
        round_decimal: int,
    ) -> None:
        """_summary_

        Args:
            mechanism_type (str): mechanism type
            experiment (str): experiment, i.e., config file in mechanism directory
            learn_alg (str): learning algorithm, i.e., config file in mechanism/learner directory
            number_runs (int): number of repetitions of experiment
            learning (bool): run learning to learn strategies
            simulation (bool): run simulation to get metrics
            logging (bool): log results
            save_strat (bool): save strategies
            path_config (str): _description_
            save_init_strat (bool, optional): path to directory containing the config files. Defaults to True.
            path_exp (str, optional): path to directory to save results. Defaults to "".
            round_decimal (int, optional): accuracy of metric in logger. Defaults to 3.
        """

        self.mechanism_type = mechanism_type
        self.experiment = experiment
        self.learn_alg = learn_alg

        self.number_runs = number_runs
        self.learning = learning
        self.simulation = simulation
        self.number_samples = number_samples

        self.save_strat = save_strat
        self.save_init_strat = save_init_strat
        self.logging = logging
        self.path_exp = path_exp

        # setup game and learner using config
        print(f"Experiment - {mechanism_type}-{experiment}-{learn_alg} - started")
        try:
            self.config = Config()
            self.config.get_path(path_config)
            self.game, self.learner = self.config.create_setting(
                mechanism_type, experiment, learn_alg
            )
            print(f" - Setting created ")
        except Exception as e:
            print(e)
            print(f" - Error: setting not created")
            self.learning, self.simulation = False, False

        # create logger
        self.logger = Logger(
            path_exp, mechanism_type, experiment, learn_alg, logging, round_decimal
        )

        # create directory to save results (strategies or log-files)
        if self.logging or self.save_strat:
            if self.path_exp == "":
                raise ValueError("path to store strategies/log-files not given")
            Path(path_exp + "strategies/" + mechanism_type).mkdir(
                parents=True, exist_ok=True
            )

    def run(self) -> None:
        """
        run experiment
        """
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

        print(
            f"Experiment - {self.mechanism_type}-{self.experiment}-{self.learn_alg} - finished\n"
        )

    def run_learning(self):
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
        ):

            # init strategies
            init_method = self.config.config_learner["init_method"]
            param_init = self.config.config_learner["param_init"]
            self.strategies = self.config.create_strategies(
                self.game, init_method, param_init
            )

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

    def run_simulation(self):
        """run simulation for computed strategies"""
        print(" - Simulation:")

        # repeat experiment
        for run in tqdm(
            range(self.number_runs),
            unit_scale=True,
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}",
            colour="blue",
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

    def save_strategies(self, run: int):
        """Save strategies for current experiment

        Args:
            run (int): current repetition of experiment
        """
        if self.save_strat:
            name = f"{self.learn_alg}_{self.experiment}_run_{run}"
            for i in self.strategies:
                self.strategies[i].save(
                    name, self.mechanism_type, self.path_exp, self.save_init_strat
                )

    def load_strategies(self, run: int):
        """Load strategies for current experiment

        Args:
            run (int): current repetition of experiment
        """
        # init strategies
        self.strategies = self.config.create_strategies(self.game)
        name = f"{self.learn_alg}_{self.experiment}_run_{run}"
        for i in self.strategies:
            self.strategies[i].load(name, self.mechanism_type, self.path_exp)

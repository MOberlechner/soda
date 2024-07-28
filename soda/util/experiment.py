import os
from pathlib import Path
from time import time

import numpy as np
from tqdm import tqdm

from soda.learner.gradient import Gradient
from soda.util.config import Config
from soda.util.logging import Logger


class Experiment:
    """
    Config class to run experiments

    Each experiment can consist of three parts:
    - learning:
        apply learning algorithm to discretized game to approximate the BNE using discrete distributional strategies
    - simulation:
        simulate continuous auctions and bid according to the computed strategies
        against other computed strategies or the analytical BNE
    - evaluation:
        evaluate computed strategies in a different discretized game, e.g., with a higher discretization.
        This is particularly useful, if we don't have an analytical BNE to compare with in the 'simulation'

    """

    def __init__(
        self,
        config_game: str,
        config_learner: str,
        number_runs: int,
        label_experiment: str = None,
        param_computation: dict = {"active": False},
        param_simulation: dict = {"active": False},
        param_evaluation: dict = {"active": False},
        param_logging: dict = {},
    ):
        """Initiliaze Experiment

        Args:
            config_game (str): config file for game/mechanism
            config_learner (str): config file for learner
            number_runs (int): number of repetitions of experiment
            label_experiment (str, optional): name for a group of different settings.
                Allows us to structure our results in different experiments. If not specified, we group the results by mechanism.
            param_computation (dict):
                - active (bool): if learning is active
                - init_strategies (str): method to initialize learning algorithm
            param_simulation (dict): parameter for simulation
                - active (bool): if simulation is active
                - number_samples (int): number of samples used for simulations
            param_evaluation (dict): parameter for evaluation
                - active (bool): if evaluation is active
                - config_game (dict): containts a config with changes for the evaluation game (i.e., higher discretization)
            param_logging (dict):
                - path_exp (str): path to directory to save results.
                - round_decimal (int, optional): accuracy of metric in logger. Defaults to 5.
                - save_strat (bool): save strategies
                - save_plots (bool): save plots of strategies
        """
        self.config_game = config_game
        self.config_learner = config_learner
        self.number_runs = number_runs
        self.label_experiment = label_experiment

        self.param_computation = param_computation
        self.param_simulation = param_simulation
        self.param_evaluation = param_evaluation
        self.param_logging = param_logging

        self.computation = param_computation["active"]
        self.simulation = param_simulation["active"]
        self.evaluation = param_evaluation["active"]
        self.error = False

        print(f"Experiment started".ljust(100, "."))
        print(f" - game:    {self.config_game}\n - learner: {self.config_learner}")

        try:
            self.prepare_experiment()
            print(f" - Setting created ")
        except Exception as e:
            print(e)
            print(f" - Error: setting not created")
            self.error = True
            self.computation, self.simulation, self.evaluation = False, False, False

    def prepare_experiment(self):
        """create setting and initialize logger"""
        # create setting (game and learner)
        self.config = Config(self.config_game, self.config_learner)
        self.game, self.learner = self.config.create_setting()

        self.label_mechanism = self.game.name
        self.label_setting = Path(self.config_game).stem
        self.label_learner = Path(self.config_learner).stem
        if self.label_experiment is None:
            self.label_experiment = self.label_mechanism

        # initialize logger
        self.logger = Logger(
            self.label_mechanism,
            self.label_setting,
            self.label_learner,
            self.label_experiment,
            self.param_logging,
        )

    # ------------------------------------ run sub-experiments ------------------------------------

    def run(self) -> None:
        """run experiment, i.e., computation, simulation, and evaluation"""
        # run learning
        if self.computation:
            try:
                self.run_compuation()
                self.logger.save("computation")
            except Exception as e:
                print(e)
                print("    Error in Computation!")
                self.error = True

        # run simulation
        if self.simulation:
            try:
                self.run_simulation()
                self.logger.save("simulation")
            except Exception as e:
                print(e)
                print("   Error in Simulation!")
                self.error = True

        # run evaluation
        if self.evaluation:
            try:
                self.run_evaluation()
                self.logger.save("evaluation")
            except Exception as e:
                print(e)
                print("   Error in Evaluation!")
                self.error = True

        print(f"Done ".ljust(100, ".") + "\n")

    def run_compuation(self) -> None:
        """run computation of strategies"""
        print(" - Computation:")
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
            init_param = (
                {}
                if "init_param" not in self.param_computation
                else self.param_learning["init_param"]
            )
            self.strategies = self.config.create_strategies(
                self.game, self.param_computation["init_method"], init_param
            )

            # run learning algorithm
            t0 = time()
            self.learner.run(self.game, self.strategies)
            time_run = time() - t0

            # log result
            self.logger.log_computation(
                strategies=self.strategies,
                learner=self.learner,
                run=run,
                time_init=time_init,
                time_run=time_run,
            )

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
            self.strategies = self.config.create_strategies(
                self.game, init_method="nan"
            )
            for agent in self.strategies:
                self.logger.load_strategies(strategy=self.strategies[agent], run=run)

            if all(self.strategies[i].x is not None for i in self.game.set_bidder):

                # sample observation and bids
                obs_profile = self.game.mechanism.sample_types(
                    self.param_simulation["number_samples"]
                )
                bid_profile = np.array(
                    [
                        self.strategies[self.game.bidder[i]].sample_bids(obs_profile[i])
                        for i in range(self.game.n_bidder)
                    ]
                )
                print("o/b sampled")
                # compute metrics for agents
                for agent in self.game.set_bidder:
                    data = self.game.mechanism.get_metrics_agents(
                        agent, obs_profile, bid_profile
                    )
                    self.logger.log_simulation(run=run, agent=agent, data=data)
                print("metrics agents computed")
                # compute metrics for mechanism
                data = self.game.mechanism.get_metrics_mechanism(
                    obs_profile, bid_profile
                )
                self.logger.log_simulation(run=run, agent="mechanism", data=data)

            else:
                break

        print(" - run_simulation finished")

    def run_evaluation(self) -> None:
        """run evaluation for computed strategies"""
        print(" - Evaluation:")

        # repeat experiment
        for run in tqdm(
            range(self.number_runs),
            unit_scale=True,
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}",
            colour="red",
            desc="    Progress",
        ):
            # create game for evaluation
            self.config_game_eval = self.get_config_game_eval()
            self.config_eval = Config(self.config_game_eval, self.config_learner)
            self.game_eval, _ = self.config_eval.create_setting()

            # load computed strategies
            self.load_strategies(run)
            self.rescale_strategies()

            # create gradient for evaluation
            gradient = Gradient()
            gradient.prepare(self.game_eval, self.strategies)

            # compute metrics
            for i in self.game_eval.set_bidder:
                grad = gradient.compute(
                    self.game_eval,
                )
                metrics, values = None, None
                for tag, val in zip(metrics, values):
                    self.logger.log_evaluation_run(run, i, tag, val)

    # ------------------------------------ helper methods for different subexperiments ------------------------------------

    def get_config_game_eval(self):
        """config game for evaluation game is identical to the original game, except for keys specified in
        the config in param_evaluation"""
        self.config_game_eval = self.config_game.copy()
        for key, val in self.param_evaluation["config"]:
            self.config_game_eval[key] = val

    def rescale_strategies(self, run: int) -> None:
        """If the loaded strategy does not fit the dimensions of the discrete game,
        we translate the strategies to the corresponding discretization"""
        raise NotImplementedError

import os
from typing import Dict

import yaml
from yaml.loader import SafeLoader

from src.game import Game
from src.learner.best_response import BestResponse
from src.learner.fictitious_play import FictitiousPlay
from src.learner.frank_wolfe import FrankWolfe
from src.learner.learner import Learner
from src.learner.soda import SODA
from src.learner.soma import SOMA
from src.mechanism.all_pay import AllPay
from src.mechanism.bertrand_pricing import BertrandPricing
from src.mechanism.contest_game import ContestGame
from src.mechanism.crowdsourcing import Crowdsourcing
from src.mechanism.double_auction import DoubleAuction
from src.mechanism.llg_auction import LLGAuction
from src.mechanism.multi_unit import MultiUnitAuction
from src.mechanism.single_item import SingleItemAuction
from src.mechanism.split_award import SplitAwardAuction
from src.strategy import Strategy


class Config:
    """
    Config Class to create mechanism, approximation game and learner
    """

    def __init__(
        self,
    ):
        """Initialize Config Class"""

    def create_experiment(self, setting: str, experiment: str, learn_alg: str):
        """create mechanism, game, strategies and learner according to config file

        Args:
            setting (str): mechanism
            experiment (str): name of config file in mechanism directory
            learn_alg (str): name of config file for learner in mechanism/learner directory
        """
        self.get_config_game(setting, experiment)
        self.get_config_learner(setting, learn_alg)

        game = self.create_game()
        learner = self.create_learner()
        return game, learner

    def get_config_game(self, setting: str, experiment: str):
        """get configuration to create mechanism and game

        Args:
            setting (str): mechanism/directory in config dir
            experiment (str): file in mechanism dir of config dir
        """
        if not hasattr(self, "path_to_config"):
            raise ValueError("Path to config not specified")

        # get config file for game
        with open(f"{self.path_to_config}{setting}/{experiment}.yaml") as f:
            config_game = yaml.load(f, Loader=SafeLoader)
        # test config file
        for key in ["bidder", "o_space", "a_space", "param_prior", "param_util"]:
            if key not in config_game:
                raise ValueError(
                    f"config file for mechanism/game is not feasible. {key} is missing."
                )
        self.config_game = config_game

    def create_config_game(
        self,
        mechanism_type: str,
        bidder: list,
        o_space: dict,
        a_space: dict,
        param_prior: dict,
        param_util: dict,
        n: int,
        m: int,
    ) -> None:
        """Create config file to create mechanism and game

        Args:
            mechanism_type (str): mechanism
            bidder (list): list of agents
            o_space (dict): contains intervals for observation space for bidders
            a_space (dict): contains intervals for action space for bidders
            param_prior (dict): contains parameters for prior
            param_util (dict): contains parameters for utility function
            n (int): number discretization points observation intervals
            m (int): number discretization points action intervals
        """
        self.config_game = {
            "mechanism": mechanism_type,
            "bidder": bidder,
            "o_space": o_space,
            "a_space": a_space,
            "param_util": param_util,
            "param_prior": param_prior,
            "n": n,
            "m": m,
        }

    def create_strategies(
        self, game: Game, init_method: str = "random", param_init: dict = {}
    ) -> Dict[str, Strategy]:
        """Create strategy profile

        Args:
            game (Game): approximation game
            init_method (str, optional): initialization method for strategy. Defaults to "random".
            param_init (dict, optional): Parameter for initialization method. Defaults to {}.

        Returns:
            dict: contains strategy for each unique bidder
        """
        strategies = {}
        for i in game.set_bidder:
            strategies[i] = Strategy(i, game)
            strategies[i].initialize(init_method, param_init)
        return strategies

    def create_game(self):
        """Create Mechanism and approximation game for configuration

        Args:
            setting (str): name of mechanism
            bidder (list): list of agents
            o_space (dict): dict with bounds for observation space
            a_space (dict): dict with bounds for action space
            param_prior (dict): dict with parameter for prior
            param_util (dict): dict with parameter for utility function

        Raises:
            ValueError: Configuration not found
            ValueError: Mechanism unknown

        Returns:
            Game, Mechanism
        """
        if not hasattr(self, "config_game"):
            raise ValueError("configuration for mechanism/game not created")

        setting = self.config_game["mechanism"]
        try:
            args = [
                self.config_game["bidder"],
                self.config_game["o_space"],
                self.config_game["a_space"],
                self.config_game["param_prior"],
                self.config_game["param_util"],
            ]
        except:
            raise ValueError("config_game doesn't contain all arguments for mechanism")

        if setting == "single_item":
            mechanism = SingleItemAuction(*args)
        elif setting == "llg_auction":
            mechanism = LLGAuction(*args)
        elif setting == "contest_game":
            mechanism = ContestGame(*args)
        elif setting == "all_pay":
            mechanism = AllPay(*args)
        elif setting == "crowdsourcing":
            mechanism = Crowdsourcing(*args)
        elif setting == "split_award":
            mechanism = SplitAwardAuction(*args)
        elif setting == "double_auction":
            mechanism = DoubleAuction(*args)
        elif setting == "bertrand_pricing":
            mechanism = BertrandPricing(*args)
        elif setting == "multi_unit":
            mechanism = MultiUnitAuction(*args)
        else:
            raise ValueError('Mechanism "{}" not available'.format(setting))

        game = Game(mechanism, self.config_game["n"], self.config_game["m"])
        return game

    def get_config_learner(self, setting: str, learn_alg: str):
        """get configuration to create mechanism and game

        Args:
            setting (str): mechanism-dir in config dir
            learn_alg (str): file in learner-dir in mechanism-dir
        """
        if not hasattr(self, "path_to_config"):
            raise ValueError("Path to config not specified")

        # get config file for learner
        with open(f"{self.path_to_config}{setting}/learner/{learn_alg}.yaml") as f:
            config_learner = yaml.load(f, Loader=SafeLoader)
        # test config file
        for key in ["name", "max_iter", "tol", "stop_criterion"]:
            if key not in config_learner:
                raise ValueError(
                    f"config file for learner is not feasible. {key} is missing."
                )
        self.config_learner = config_learner

    def create_config_learner(
        self,
        learner_type: str,
        max_iter: int,
        tol: float,
        stop_criterion: str,
        param: Dict = {},
    ):
        """create config file to create learn algorithm

        Args:
            learn_alg (str): name of learn algorithm
            max_iter (int): max number of iterations
            tol (float): tolerance for stop criterion
            stop_criteration (str): stop criterion (e.g. util_loss)
            method (str, optional): specify method for learner (e.g. mirror map, regularizer,...). Defaults to None.
            steprule_bool (bool, optional): use step rule. Defaults to None.
            eta (float, optional): parameter1 for steprule. Defaults to None.
            beta (float, optional): parameter2 for steprule. Defaults to None.
        """
        self.config_learner = {
            "learner": learner_type,
            "max_iter": max_iter,
            "tol": tol,
            "stop_criterion": stop_criterion,
            "parameter": param,
        }

    def create_learner(self) -> Learner:
        """create learn algorithm from learner configurationss

        Raises:
            ValueError: config file not available
            ValueError: learner unknown

        Returns:
            Learner: learn algorithm
        """
        if not hasattr(self, "config_learner"):
            raise ValueError("configuration for learner not created")

        learn_alg = self.config_learner["learner"]
        args = [
            self.config_learner["max_iter"],
            self.config_learner["tol"],
            self.config_learner["stop_criterion"],
        ]
        param = self.config_learner["parameter"]

        if learn_alg == "soda":
            return SODA(*args, param)

        elif learn_alg == "soma":
            return SOMA(*args, param)

        elif learn_alg == "frank_wolfe":
            return FrankWolfe(*args, param)

        elif learn_alg == "fictitious_play":
            return FictitiousPlay(*args, param)

        elif learn_alg == "best_response":
            return BestResponse(*args, param)

        else:
            raise ValueError(f"Learner {learn_alg} unknown.")

    def get_path(self, path):
        """get path into config directory

        Args:
            path (str):
        """
        if path is not None:
            self.path_into_project = os.getcwd().split("soda")[0] + "soda/"
            self.path_to_config = f"{self.path_into_project}{path}/".replace("//", "/")

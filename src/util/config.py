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
from src.mechanism.all_pay import AllPayAuction
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
    Config Class to create mechanism, approximation game and learner.
    This class allows us to either import configuration files to create the respective game, learner and strategies
    or to enter these specifications manually. This can be used to run experiments in notebooks, or to run experiments.

    Attributes:
        config_game                 dict: either imported or created, contains all relevant parameters to create
                                    a mechanism and the respective approximation game
        config_leaner               dict: either imported or created, contains all relevant parameter to create
                                    a learner and the initial strategies
        path_to_config (optional)   str: using the get_path() method with the path from soda/ to the config directory
                                    the attribute is created to import config files from there

    Methods:
        create_setting()            imports config files from path_to_config/mechanism_type/
                                    and creates setting, i.e., game and learner, we want to consider.
    """

    def __init__(
        self,
    ):
        """Initialize Config Class"""

    def create_setting(self, mechanism_type: str, experiment: str, learn_alg: str):
        """create game and learner according to config file.

        Args:
            mechanism_type (str): mechanism type
            experiment (str): name of config file in mechanism directory
            learn_alg (str): name of config file for learner in mechanism/learner directory
        """
        self.get_config_game(mechanism_type, experiment)
        self.get_config_learner(mechanism_type, learn_alg)

        game = self.create_game()
        learner = self.create_learner()

        return game, learner

    def get_config_game(self, mechanism_type: str, experiment: str):
        """get configuration to create mechanism and game

        Args:
            setting (str): mechanism/directory in config dir
            experiment (str): file in mechanism dir of config dir
        """
        if not hasattr(self, "path_to_config"):
            raise ValueError("Path to config not specified")

        # get config file for game
        try:
            with open(f"{self.path_to_config}{mechanism_type}/{experiment}.yaml") as f:
                config_game = yaml.load(f, Loader=SafeLoader)
        except FileNotFoundError:
            raise ValueError(
                f"couldn't open: {self.path_to_config}{mechanism_type}/{experiment}.yaml"
            )

        # test config file
        for key in ["bidder", "o_space", "a_space", "param_prior", "param_util"]:
            if key not in config_game:
                raise ValueError(
                    f"config file for mechanism/game is not feasible. {key} is missing."
                )
        self.config_game = config_game
        self.config_game["mechanism_type"] = mechanism_type

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
            "mechanism_type": mechanism_type,
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

        mechanism_type = self.config_game["mechanism_type"]
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

        if mechanism_type == "single_item":
            mechanism = SingleItemAuction(*args)
        elif mechanism_type == "llg_auction":
            mechanism = LLGAuction(*args)
        elif mechanism_type == "contest_game":
            mechanism = ContestGame(*args)
        elif mechanism_type == "all_pay":
            mechanism = AllPayAuction(*args)
        elif mechanism_type == "crowdsourcing":
            mechanism = Crowdsourcing(*args)
        elif mechanism_type == "split_award":
            mechanism = SplitAwardAuction(*args)
        elif mechanism_type == "double_auction":
            mechanism = DoubleAuction(*args)
        elif mechanism_type == "bertrand_pricing":
            mechanism = BertrandPricing(*args)
        elif mechanism_type == "multi_unit":
            mechanism = MultiUnitAuction(*args)
        else:
            raise ValueError(f"Mechanism {mechanism_type} not available")

        try:
            n = self.config_game["n"]
            m = self.config_game["m"]
        except:
            raise ValueError("config_game doesn't contain parameter for discretization")

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
        # test config file for learner
        for key in ["learn_alg", "max_iter", "tol", "stop_criterion", "init_method"]:
            if key not in config_learner:
                raise ValueError(
                    f"config file for learner is not feasible. {key} is missing."
                )
        # test config file for parameter (learner or init)
        if "parameter" not in config_learner:
            config_learner["parameter"] = {}
        if "param_init" not in config_learner:
            config_learner["param_init"] = {}

        self.config_learner = config_learner

    def create_config_learner(
        self,
        learn_alg: str,
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
            "learn_alg": learn_alg,
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

        learn_alg = self.config_learner["learn_alg"]
        args = [
            self.config_learner["max_iter"],
            float(self.config_learner["tol"]),
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

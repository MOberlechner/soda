import os

import yaml
from yaml.loader import SafeLoader

from src.game import Game
from src.learner.best_response import BestResponse
from src.learner.fictitious_play import FictitiousPlay
from src.learner.frank_wolfe import FrankWolfe
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

    def create_experiment(setting: str, experiment: str, learner: str):
        """create mechanism, game, strategies and learner according to config file

        Args:
            setting (str): mechanism
            experiment (str): name of config file in mechanism directory
            learner (str): name of config file for learner in mechanism/learner directory
        """
        pass

    def create_config_game(
        self,
        setting: str,
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
            setting (str): mechanism
            bidder (list): list of agents
            o_space (dict): contains intervals for observation space for bidders
            a_space (dict): contains intervals for action space for bidders
            param_prior (dict): contains parameters for prior
            param_util (dict): contains parameters for utility function
            n (int): number discretization points observation intervals
            m (int): number discretization points action intervals
        """
        self.config_game = {
            "mechanism": setting,
            "bidder": bidder,
            "o_space": o_space,
            "a_space": a_space,
            "param_util": param_util,
            "param_prior": param_prior,
            "n": n,
            "m": m,
        }

    def create_strategies(
        self, game, init_method: str = "random", param_init: dict = {}
    ) -> dict:
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

    def create_game_mechanism(self):
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
        args = [
            self.config_game["bidder"],
            self.config_game["o_space"],
            self.config_game["a_space"],
            self.config_game["param_prior"],
            self.config_game["param_util"],
        ]

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
        return game, mechanism

    def create_config_learner(
        self,
        learn_alg: str,
        max_iter: int,
        tol: float,
        stop_criteration: str,
        method: str = None,
        steprule_bool: bool = None,
        eta: float = None,
        beta: float = None,
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
            "learner": learn_alg,
            "max_iter": max_iter,
            "tol": tol,
            "stop_criteration": stop_criteration,
        }
        if method is not None:
            self.config_learner["method"] = method
        if steprule_bool is not None:
            self.config_learner["steprule_bool"] = steprule_bool
        if eta is not None:
            self.config_learner["eta"] = eta
        if beta is not None:
            self.config_learner["beta"] = beta

    def create_learner(self):
        """create learn algorithm from learner configurations"""
        if not hasattr(self, "config_learner"):
            raise ValueError("configuration for learner not created")

        learn_alg = self.config_learner["learner"]
        if learn_alg == "soda":
            return SODA(
                self.config_learner["max_iter"],
                self.config_learner["tol"],
                self.config_learner["stop_criterion"],
                self.config_learner["regularizer"],
                self.config_learner["steprule_bool"],
                self.config_learner["eta"],
                self.config_learner["beta"],
            )

        elif learn_alg == "soma":
            return SOMA(
                self.config_learner["max_iter"],
                self.config_learner["tol"],
                self.config_learner["stop_criterion"],
                self.config_learner["mirror_map"],
                self.config_learner["steprule_bool"],
                self.config_learner["eta"],
                self.config_learner["beta"],
            )

        elif learn_alg == "frank_wolfe":
            return FrankWolfe(
                self.config_learner["max_iter"],
                self.config_learner["tol"],
                self.config_learner["stop_criterion"],
                self.config_learner["method"],
                self.config_learner["steprule_bool"],
                self.config_learner["eta"],
                self.config_learner["beta"],
            )

        elif learn_alg == "fictitious_play":
            return FictitiousPlay(
                self.config_learner["max_iter"],
                self.config_learner["tol"],
                self.config_learner["stop_criterion"],
            )

        elif learn_alg == "best_response":
            return BestResponse(
                self.config_learner["max_iter"],
                self.config_learner["tol"],
                self.config_learner["stop_criterion"],
            )
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

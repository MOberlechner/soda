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

    def create_game(self, mechanism, n: int, m: int):
        """Create approximation game given the mechanism

        Args:
            mechanism (Mechanism):
            n (int): discretization points type space
            m (int): discretization points action space

        Returns:
            Game:
        """
        return Game(mechanism, n, m)

    def create_mechanism(
        self,
    ):
        """_summary_

        Args:
            setting (str): name of mechanism
            bidder (list): list of agents
            o_space (dict): dict with bounds for observation space
            a_space (dict): dict with bounds for action space
            param_prior (dict): dict with parameter for prior
            param_util (dict): dict with parameter for utility function

        Raises:
            ValueError: Mechanism unknown

        Returns:
            Mechanism
        """
        if not hasattr(self, "config_game"):
            raise ValueError("configuration for mechanism/game not created")

        setting = self.config_game["mechanism"]

        if setting == "single_item":
            mechanism = SingleItemAuction(
                self.config_game["bidder"],
                self.config_game["o_space"],
                self.config_game["a_space"],
                self.config_game["param_prior"],
                self.config_game["param_util"],
            )
        elif setting == "llg_auction":
            mechanism = LLGAuction(
                self.config_game["bidder"],
                self.config_game["o_space"],
                self.config_game["a_space"],
                self.config_game["param_prior"],
                self.config_game["param_util"],
            )
        elif setting == "contest_game":
            mechanism = ContestGame(
                self.config_game["bidder"],
                self.config_game["o_space"],
                self.config_game["a_space"],
                self.config_game["param_prior"],
                self.config_game["param_util"],
            )
        elif setting == "all_pay":
            mechanism = AllPay(
                self.config_game["bidder"],
                self.config_game["o_space"],
                self.config_game["a_space"],
                self.config_game["param_prior"],
                self.config_game["param_util"],
            )
        elif setting == "crowdsourcing":
            mechanism = Crowdsourcing(
                self.config_game["bidder"],
                self.config_game["o_space"],
                self.config_game["a_space"],
                self.config_game["param_prior"],
                self.config_game["param_util"],
            )
        elif setting == "split_award":
            mechanism = SplitAwardAuction(
                self.config_game["bidder"],
                self.config_game["o_space"],
                self.config_game["a_space"],
                self.config_game["param_prior"],
                self.config_game["param_util"],
            )
        elif setting == "double_auction":
            mechanism = DoubleAuction(
                self.config_game["bidder"],
                self.config_game["o_space"],
                self.config_game["a_space"],
                self.config_game["param_prior"],
                self.config_game["param_util"],
            )
        elif setting == "bertrand_pricing":
            mechanism = BertrandPricing(
                self.config_game["bidder"],
                self.config_game["o_space"],
                self.config_game["a_space"],
                self.config_game["param_prior"],
                self.config_game["param_util"],
            )
        elif setting == "multi_unit":
            mechanism = MultiUnitAuction(
                self.config_game["bidder"],
                self.config_game["o_space"],
                self.config_game["a_space"],
                self.config_game["param_prior"],
                self.config_game["param_util"],
            )
        else:
            raise ValueError('Mechanism "{}" not available'.format(setting))
        return mechanism

    def create_learner(
        self,
        name: str,
        max_iter: int,
        tol: float,
        stop_criterion: str,
        regularizer: str = None,
        mirror_map: str = None,
        method: str = None,
        step_rule: bool = True,
        eta: float = None,
        beta: float = None,
    ):
        """_summary_

        Args:
            max_iter (int): _description_
            tol (float): _description_
            stop_criteration (str): _description_
            regularizer (str, optional): _description_. Defaults to None.
            mirror_map (str, optional): _description_. Defaults to None.
            step_rule (bool, optional): _description_. Defaults to True.
            eta (float, optional): _description_. Defaults to None.
            beta (float, optional): _description_. Defaults to None.
        """
        if name == "soda":
            return SODA(
                max_iter,
                tol,
                stop_criterion,
                regularizer,
                step_rule,
                eta,
                beta,
            )

        elif name == "soma":
            return SOMA(
                max_iter,
                tol,
                stop_criterion,
                mirror_map,
                step_rule,
                eta,
                beta,
            )

        elif name == "frank_wolfe":
            return FrankWolfe(
                max_iter,
                tol,
                stop_criterion,
                method,
                step_rule,
                eta,
                beta,
            )

        elif name == "fictitious_play":
            return FictitiousPlay(
                max_iter,
                tol,
                stop_criterion,
            )

        elif name == "best_response":
            return BestResponse(
                max_iter,
                tol,
                stop_criterion,
            )

        else:
            raise ValueError(f"Learner {name} unknown.")

    def get_path(self, path):
        """get path into config directory

        Args:
            path (str):
        """
        if path is not None:
            self.path_into_project = os.getcwd().split("soda")[0] + "soda/"
            self.path_to_config = f"{self.path_into_project}{path}/".replace("//", "/")

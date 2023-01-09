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

    def create_strategies(
        self, game, init_method: str = "random", param_init: dict = {}
    ) -> dict:
        """create dict with all strategies

        Args:
            game (Game): approximation game
            init_method:
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
        setting: str,
        bidder: list,
        o_space: dict,
        a_space: dict,
        param_prior: dict,
        param_util: dict,
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

        if setting == "single_item":
            mechanism = SingleItemAuction(
                bidder,
                o_space,
                a_space,
                param_prior,
                param_util,
            )
        elif setting == "llg_auction":
            mechanism = LLGAuction(
                bidder,
                o_space,
                a_space,
                param_prior,
                param_util,
            )
        elif self.setting == "contest_game":
            mechanism = ContestGame(
                bidder,
                o_space,
                a_space,
                param_prior,
                param_util,
            )
        elif self.setting == "all_pay":
            mechanism = AllPay(
                bidder,
                o_space,
                a_space,
                param_prior,
                param_util,
            )
        elif self.setting == "crowdsourcing":
            mechanism = Crowdsourcing(
                bidder,
                o_space,
                a_space,
                param_prior,
                param_util,
            )
        elif self.setting == "split_award":
            mechanism = SplitAwardAuction(
                bidder,
                o_space,
                a_space,
                param_prior,
                param_util,
            )
        elif self.setting == "double_auction":
            mechanism = DoubleAuction(
                bidder,
                o_space,
                a_space,
                param_prior,
                param_util,
            )
        elif self.setting == "bertrand_pricing":
            mechanism = BertrandPricing(
                bidder,
                o_space,
                a_space,
                param_prior,
                param_util,
            )
        else:
            raise ValueError('Mechanism "{}" not available'.format(setting))

        return mechanism

    def create_learner(
        self,
        max_iter: int,
        tol: float,
        stop_criteration: str,
        regularizer: str = None,
        mirror_map: str = None,
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
        # todo

    def get_path(self, path):
        """get path into config directory

        Args:
            path (str):
        """
        if path is not None:
            self.path_into_project = os.getcwd().split("soda")[0] + "soda/"
            self.path_to_config = f"{self.path_into_project}{path}/".replace("//", "/")

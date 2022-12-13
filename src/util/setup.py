import os

import yaml
from yaml.loader import SafeLoader

from src.game import Game
from src.learner.best_response import BestResponse
from src.learner.frank_wolfe import FrankWolfe
from src.learner.soda import SODA
from src.learner.soma import SOMA
from src.mechanism.all_pay import AllPay
from src.mechanism.contest_game import ContestGame
from src.mechanism.crowdsourcing import Crowdsourcing
from src.mechanism.double_auction import DoubleAuction
from src.mechanism.llg_auction import LLGAuction
from src.mechanism.single_item import SingleItemAuction
from src.mechanism.split_award import SplitAwardAuction


def create_setting(setting: str, cfg):
    """Initialize mechanism, discretized game and strategies

    Args:
        setting (str): specifies mechanism
        cfg (_type_): config file

    Raises:
        ValueError: setting unknown

    Returns:
        tuple: mechanism, game, strategy
    """

    if setting == "single_item":
        mechanism = SingleItemAuction(
            cfg["bidder"],
            cfg["o_space"],
            cfg["a_space"],
            cfg["param_prior"],
            cfg["param_util"],
        )

    elif setting == "llg_auction":
        mechanism = LLGAuction(
            cfg["bidder"],
            cfg["o_space"],
            cfg["a_space"],
            cfg["param_prior"],
            cfg["param_util"],
        )

    elif setting == "contest_game":
        mechanism = ContestGame(
            cfg["bidder"],
            cfg["o_space"],
            cfg["a_space"],
            cfg["param_prior"],
            cfg["param_util"],
        )
    elif setting == "all_pay":
        mechanism = AllPay(
            cfg["bidder"],
            cfg["o_space"],
            cfg["a_space"],
            cfg["param_prior"],
            cfg["param_util"],
        )

    elif setting == "crowdsourcing":
        mechanism = Crowdsourcing(
            cfg["bidder"],
            cfg["o_space"],
            cfg["a_space"],
            cfg["param_prior"],
            cfg["param_util"],
        )

    elif setting == "split_award":
        mechanism = SplitAwardAuction(
            cfg["bidder"],
            cfg["o_space"],
            cfg["a_space"],
            cfg["param_prior"],
            cfg["param_util"],
        )

    elif setting == "double_auction":
        mechanism = DoubleAuction(
            cfg["bidder"],
            cfg["o_space"],
            cfg["a_space"],
            cfg["param_prior"],
            cfg["param_util"],
        )

    else:
        raise ValueError('Mechanism "{}" not available'.format(setting))

    # create approximation game
    game = Game(mechanism, cfg["n"], cfg["m"])

    return mechanism, game


def create_learner(cfg_learner):
    """Initialize learner

    Parameters
    ----------
    cfg_learner : config for learning algorithm

    Returns
    -------
    learner
    """
    if cfg_learner["name"] == "soda":
        return SODA(
            cfg_learner["max_iter"],
            cfg_learner["tol"],
            cfg_learner["stop_criterion"],
            cfg_learner["regularizer"],
            cfg_learner["steprule_bool"],
            cfg_learner["eta"],
            cfg_learner["beta"],
        )

    elif cfg_learner["name"] == "soma":
        return SOMA(
            cfg_learner["max_iter"],
            cfg_learner["tol"],
            cfg_learner["stop_criterion"],
            cfg_learner["mirror_map"],
            cfg_learner["steprule_bool"],
            cfg_learner["eta"],
            cfg_learner["beta"],
        )

    elif cfg_learner["name"] == "frank_wolfe":
        return FrankWolfe(
            cfg_learner["max_iter"],
            cfg_learner["tol"],
            cfg_learner["stop_criterion"],
            cfg_learner["method"],
            cfg_learner["steprule_bool"],
            cfg_learner["eta"],
            cfg_learner["beta"],
        )

    elif cfg_learner["name"] == "best_response":
        return BestResponse(
            cfg_learner["max_iter"],
            cfg_learner["tol"],
            cfg_learner["stop_criterion"],
        )

    else:
        raise ValueError("Learner {} unknown.".format(cfg_learner["name"]))


def get_config(path_config: str, setting: str, experiment: str, learn_alg: str):
    """Get config file

    Args:
        path_config (str): path to config directory
        setting (str): subdirectory for mechanism
        experiment (str): experiment for mechanism (setting)
        learn_alg (str): learning algorithm in directory setting/learner

    Returns:
        config files for experiment and learner
    """
    # current directory:
    current_dir = os.getcwd()
    # path to project
    dir_project = current_dir.split("soda")[0] + "soda/"
    # path from soda into configs
    path_config = f"{path_config}/".replace("//", "/")

    # get auction game
    with open(f"{dir_project}{path_config}{setting}/{experiment}.yaml") as f:
        cfg_exp = yaml.load(f, Loader=SafeLoader)

    # get learner
    with open(f"{dir_project}{path_config}{setting}/learner/{learn_alg}.yaml") as f:
        cfg_learner = yaml.load(f, Loader=SafeLoader)

    # test file
    if any(
        key not in cfg_exp
        for key in ["bidder", "o_space", "a_space", "param_prior", "param_util"]
    ):
        raise ValueError(
            "config file for mechanism/game is not feasible. key is missing."
        )
    elif experiment != cfg_exp["name"]:
        raise ValueError("name in config file does not coincide with name of the file")
    elif isinstance(cfg_learner["tol"], str):
        try:
            cfg_learner["tol"] = float(cfg_learner["tol"])
        except:
            raise ValueError("tol in config learner is not a float")

    return cfg_exp, cfg_learner

from src.game import Game
from src.learner.best_response import BestResponse
from src.learner.frank_wolfe import FrankWolfe
from src.learner.poga import POGA
from src.learner.soda import SODA
from src.mechanism.all_pay import AllPay
from src.mechanism.contest_game import ContestGame
from src.mechanism.crowdsourcing import Crowdsourcing
from src.mechanism.double_auction import DoubleAuction
from src.mechanism.llg_auction import LLGAuction
from src.mechanism.single_item import SingleItemAuction
from src.mechanism.split_award import SplitAwardAuction
from src.strategy import Strategy


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
            cfg.bidder, cfg.o_space, cfg.a_space, cfg.param_prior, cfg.param_util
        )

    elif setting == "llg_auction":
        mechanism = LLGAuction(
            cfg.bidder, cfg.o_space, cfg.a_space, cfg.param_prior, cfg.param_util
        )

    elif setting == "contest_game":
        mechanism = ContestGame(
            cfg.bidder, cfg.o_space, cfg.a_space, cfg.param_prior, cfg.param_util
        )
    elif setting == "all_pay":
        mechanism = AllPay(
            cfg.bidder, cfg.o_space, cfg.a_space, cfg.param_prior, cfg.param_util
        )

    elif setting == "crowdsourcing":
        mechanism = Crowdsourcing(
            cfg.bidder, cfg.o_space, cfg.a_space, cfg.param_prior, cfg.param_util
        )

    elif setting == "split_award":
        mechanism = SplitAwardAuction(
            cfg.bidder, cfg.o_space, cfg.a_space, cfg.param_prior, cfg.param_util
        )

    elif setting == "double_auction":
        mechanism = DoubleAuction(
            cfg.bidder, cfg.o_space, cfg.a_space, cfg.param_prior, cfg.param_util
        )

    else:
        raise ValueError('Mechanism "{}" not available'.format(setting))

    # create approximation game
    game = Game(mechanism, cfg.n, cfg.m)

    # create strategies
    strategies = {}
    for i in game.set_bidder:
        strategies[i] = Strategy(i, game)

    return mechanism, game, strategies


def create_learner(cfg_learner):
    """Initialize learner

    Parameters
    ----------
    cfg_learner : config for learning algorithm

    Returns
    -------
    learner
    """
    if cfg_learner.name == "soda":
        return SODA(
            cfg_learner.max_iter,
            cfg_learner.tol,
            cfg_learner.steprule_bool,
            cfg_learner.eta,
            cfg_learner.beta,
        )

    elif cfg_learner.name == "poga":
        return POGA(
            cfg_learner.max_iter,
            cfg_learner.tol,
            cfg_learner.steprule_bool,
            cfg_learner.eta,
            cfg_learner.beta,
        )

    elif cfg_learner.name == "frank_wolfe":
        return FrankWolfe(
            cfg_learner.max_iter,
            cfg_learner.tol,
        )

    elif cfg_learner.name == "best_response":
        return BestResponse(
            cfg_learner.max_iter,
            cfg_learner.tol,
        )

    else:
        raise ValueError("Learner {} unknown.".format(cfg_learner.name))

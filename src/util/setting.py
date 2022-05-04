from src.game import Game
from src.mechanism.all_pay import AllPay
from src.mechanism.contest_game import ContestGame
from src.mechanism.crowdsourcing import Crowdsourcing
from src.mechanism.llg_auction import LLGAuction
from src.mechanism.single_item import SingleItemAuction
from src.strategy import Strategy


def create_setting(setting: str, cfg):
    """Initialize mechanism, discretized game and strategies

    Parameters
    ----------
    setting : str, specifies mechanism
    cfg : config for experimt

    Returns
    -------
    mechanism, game, strategy
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
            cfg.bidder,
            cfg.o_space,
            cfg.a_space,
            cfg.param_prior,
            cfg.csf,
            cfg.param_csf,
        )
    elif setting == "all_pay":
        mechanism = AllPay(
            cfg.bidder,
            cfg.o_space,
            cfg.a_space,
            cfg.param_prior,
            cfg.util_setting,
            cfg.param_util,
        )

    elif setting == "crowdsourcing":
        mechanism = Crowdsourcing(
            cfg.bidder,
            cfg.o_space,
            cfg.a_space,
            cfg.param_prior,
            cfg.price,
            cfg.param_util,
        )

    else:
        raise ValueError('Mechanism "' + setting + '" not available')

    # create approximation game
    game = Game(mechanism, cfg.n, cfg.m)

    # create and initialize strategies
    strategies = {}
    for i in game.set_bidder:
        strategies[i] = Strategy(i, game)

    return mechanism, game, strategies

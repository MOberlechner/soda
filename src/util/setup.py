import hydra

from src.game import Game
from src.learner.best_response import BestResponse
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

    elif setting == "bertrand_pricing":
        mechanism = BertrandPricing(
            cfg.bidder, cfg.o_space, cfg.a_space, cfg.param_prior, cfg.param_util
        )

    else:
        raise ValueError('Mechanism "{}" not available'.format(setting))

    # create approximation game
    game = Game(mechanism, cfg.n, cfg.m)

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
    if cfg_learner.name == "soda":
        return SODA(
            cfg_learner.max_iter,
            cfg_learner.tol,
            cfg_learner.stop_criterion,
            cfg_learner.regularizer,
            cfg_learner.steprule_bool,
            cfg_learner.eta,
            cfg_learner.beta,
        )

    elif cfg_learner.name == "soma":
        return SOMA(
            cfg_learner.max_iter,
            cfg_learner.tol,
            cfg_learner.stop_criterion,
            cfg_learner.mirror_map,
            cfg_learner.steprule_bool,
            cfg_learner.eta,
            cfg_learner.beta,
        )

    elif cfg_learner.name == "frank_wolfe":
        return FrankWolfe(
            cfg_learner.max_iter,
            cfg_learner.tol,
            cfg_learner.stop_criterion,
            cfg_learner.method,
            cfg_learner.steprule_bool,
            cfg_learner.eta,
            cfg_learner.beta,
        )

    elif cfg_learner.name == "best_response":
        return BestResponse(
            cfg_learner.max_iter,
            cfg_learner.tol,
            cfg_learner.stop_criterion,
        )

    else:
        raise ValueError("Learner {} unknown.".format(cfg_learner.name))


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
    # add this s.t. we start in the project directory, i.e. path_config should be "configs" or "data_paper/publication/configs"
    path_config = "../../" + path_config

    # get auction game
    hydra.initialize(config_path=path_config + setting, job_name="run")
    cfg_exp = hydra.compose(config_name=experiment)
    hydra.core.global_hydra.GlobalHydra().clear()

    # get learner
    hydra.initialize(config_path=path_config + setting + "/learner", job_name="run")
    cfg_learner = hydra.compose(config_name=learn_alg)
    hydra.core.global_hydra.GlobalHydra().clear()

    return cfg_exp, cfg_learner

from pathlib import Path
from time import time

import hydra

from src.learner.frank_wolfe import FrankWolfe
from src.learner.soda import SODA
from src.util.logging import log_run
from src.util.setting import create_setting


def learn_strategies(mechanism, game, strategies, cfg_learner) -> None:
    """Runs Learner to compute strategies given the specified setting.

    Args:
        mechanism: continuous auction game
        game: approximation game
        strategies: dict of strategies
        cfg_learner: parameter for Learning Algorithm
    """

    # init learner
    if cfg_learner.name == "soda":
        learner = SODA(
            cfg_learner.max_iter,
            cfg_learner.tol,
            cfg_learner.steprule_bool,
            cfg_learner.eta,
            cfg_learner.beta,
        )
    elif cfg_learner.name == "frank_wolfe":
        learner = FrankWolfe(
            cfg_learner.max_iter,
            cfg_learner.tol,
            cfg_learner.steprule_bool,
        )
    else:
        raise ValueError("Learner {} unknown.".format(cfg_learner.name))

    # initialize strategies
    init_method = cfg_learner.init_method if "init" in cfg_learner else "random"
    for i in game.set_bidder:
        strategies[i].initialize(init_method)

    # run soda
    learner.run(mechanism, game, strategies)

    return strategies


def run_experiment(learn_alg, setting, experiment, logging, runs, path, path_config):
    """
    Run experiments specified in experiments_list (only single setting possible)
    with specified learning algorithm.
    """

    # prepare stuff
    hydra.initialize(config_path=path_config + setting, job_name="run")
    Path(path + "strategies/" + setting).mkdir(parents=True, exist_ok=True)

    # get parameter
    cfg_learner = hydra.compose(config_name=learn_alg)
    cfg = hydra.compose(config_name=experiment)

    # initialize setting and compute utility
    t0 = time()
    mechanism, game, strategies = create_setting(setting, cfg)
    if not mechanism.own_gradient:
        print('Computations of Utilities for experiment: "' + experiment + '" started!')
        game.get_utility(mechanism)
    time_init = time() - t0

    # run soda
    for r in range(runs):
        t0 = time()
        strategies = learn_strategies(mechanism, game, strategies, cfg_learner)
        time_run = time() - t0

        # log and save
        if logging is True:
            log_run(
                strategies,
                cfg_learner,
                experiment,
                setting,
                r,
                time_init,
                time_run,
                path,
            )

            for i in game.set_bidder:
                name = (
                    cfg_learner.name
                    + "_"
                    + experiment
                    + ("_run_" + str(r) if runs > 1 else "")
                )
                strategies[i].save(name, path)

    print('Experiment: "' + experiment + '" finished!')


# -------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":

    learn_alg = "frank_wolfe"
    setting = "llg_auction"
    experiments_list = [
        "llg_auction_nb_gamma1",
        "llg_auction_nb_gamma2",
        "llg_auction_nb_gamma3",
        "llg_auction_nvcg_gamma1",
        "llg_auction_nvcg_gamma2",
        "llg_auction_nvcg_gamma3",
        "llg_auction_nz_gamma1",
        "llg_auction_nz_gamma2",
        "llg_auction_nz_gamma3",
    ]
    logging = True
    runs = 10
    path = "experiment/"
    path_config = "configs/"

    for experiment in experiments_list:
        run_experiment(learn_alg, setting, experiment, logging, runs, path)

import sys
from pathlib import Path
from time import time

from tqdm import tqdm

sys.path.append("../../")

from src.strategy import Strategy
from src.util.logging import Logger
from src.util.run import *


def learn_strategies(mechanism, game, logger, cfg_learner, eta, beta):
    """Runs Learner to compute strategies given the specified setting.

    Args:
        mechanism: continuous auction game
        game: approximation game
        cfg_learner: parameter for Learning Algorithm
    """

    # init learner
    learner = create_learner(cfg_learner)

    # change parameter
    learner.eta = eta
    learner.beta = beta
    logger.learn_alg = cfg_learner.name + f"_{eta}_{beta}"

    # initialize strategies
    init_method = cfg_learner.init_method if "init_method" in cfg_learner else "random"
    strategies = {}
    for i in game.set_bidder:
        strategies[i] = Strategy(i, game)
        strategies[i].initialize(init_method)

    # run soda
    learner.run(mechanism, game, strategies)
    return strategies, learner.convergence


def test_hyperparameter(setting, experiment, learn_alg, etas, betas):

    # directory to store results
    Path(path + "strategies/" + setting).mkdir(parents=True, exist_ok=True)
    logger = Logger(path, setting, experiment, learn_alg, logging, round_decimal=5)

    # get parameter
    cfg, cfg_learner = get_config(path_config, setting, experiment, learn_alg)

    # initialize setting and compute utility
    t0 = time()
    mechanism, game = create_setting(setting, cfg)
    if not mechanism.own_gradient:
        print('Computations of Utilities for experiment: "' + experiment + '" started!')
        game.get_utility(mechanism)
        print(
            'Computations of Utilities for experiment: "' + experiment + '" finished!'
        )
    time_init = time() - t0

    for eta in etas:
        for beta in betas:
            print(f"eta={eta}, beta={beta}")

            for run in tqdm(
                range(num_runs),
                unit_scale=True,
                bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}",
            ):
                t0 = time()
                # learn stratege
                strategies, convergence = learn_strategies(
                    mechanism, game, logger, cfg_learner, eta, beta
                )
                time_run = time() - t0

                # log and save
                logger.log_learning(strategies, run, convergence, time_init, time_run)

            logger.log_experiment_learning()


if __name__ == "__main__":

    path_config = "experiment/soda_or/configs/"
    path = ""
    setting = "single_item"
    experiment = "affiliated_values"
    learn_alg = "soda_entro"

    # computation
    learning = True
    num_runs = 5

    etas = [1, 2, 10, 20, 50]
    betas = [0.95, 0.50]

    # simulation
    simulation = False
    logging = True

    test_hyperparameter(setting, experiment, learn_alg, etas, betas)

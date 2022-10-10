from pathlib import Path
from time import time

from src.util.logging import log_run, log_strat
from src.util.setup import create_learner, create_setting, get_config


def learn_strategies(mechanism, game, strategies, cfg_learner) -> None:
    """Runs Learner to compute strategies given the specified setting.

    Args:
        mechanism: continuous auction game
        game: approximation game
        strategies: dict of strategies
        cfg_learner: parameter for Learning Algorithm
    """

    # init learner
    learner = create_learner(cfg_learner)

    # initialize strategies
    init_method = cfg_learner.init_method if "init" in cfg_learner else "random"
    for i in game.set_bidder:
        strategies[i].initialize(init_method)

    # run soda
    learner.run(mechanism, game, strategies)

    return strategies


def run_experiment(
    learn_alg, setting, experiment, logging, num_runs, path, path_config
):
    """
    Run experiments specified in experiments_list (only single setting possible)
    with specified learning algorithm.
    """
    # directory to store results
    Path(path + "strategies/" + setting).mkdir(parents=True, exist_ok=True)

    # get parameter
    cfg, cfg_learner = get_config(path_config, setting, experiment, learn_alg)

    # initialize setting and compute utility
    t0 = time()
    mechanism, game, strategies = create_setting(setting, cfg)
    if not mechanism.own_gradient:
        print('Computations of Utilities for experiment: "' + experiment + '" started!')
        game.get_utility(mechanism)
    time_init = time() - t0

    # run soda
    for run in range(num_runs):
        t0 = time()
        strategies = learn_strategies(mechanism, game, strategies, cfg_learner)
        time_run = time() - t0

        # log and save
        if logging is True:
            log_strat(
                strategies,
                cfg_learner,
                learn_alg,
                experiment,
                setting,
                run,
                time_init,
                time_run,
                path,
            )

            log_run(
                strategies,
                learn_alg,
                experiment,
                setting,
                run,
                path,
            )

            for i in game.set_bidder:
                name = (
                    cfg_learner.name
                    + "_"
                    + experiment
                    + ("_run_" + str(run) if num_runs > 1 else "")
                )
                # save strategies
                strategies[i].save(name, setting, path, save_init=True)
    print('Experiment: "' + experiment + '" finished!')


# -------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":

    learn_alg = "frank_wolfe"
    setting = "single_item"
    experiments_list = [
        "fpsb",
    ]
    logging = True
    runs = 10
    path = "experiment/"
    path_config = "configs/"

    for experiment in experiments_list:
        run_experiment(learn_alg, setting, experiment, logging, runs, path)

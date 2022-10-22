from pathlib import Path
from time import time

from tqdm import tqdm

from src.strategy import Strategy
from src.util.logging import log_run, log_strat
from src.util.setup import create_learner, create_setting, get_config


def learn_strategies(mechanism, game, cfg_learner) -> None:
    """Runs Learner to compute strategies given the specified setting.

    Args:
        mechanism: continuous auction game
        game: approximation game
        cfg_learner: parameter for Learning Algorithm
    """

    # init learner
    learner = create_learner(cfg_learner)

    # initialize strategies
    init_method = cfg_learner.init_method if "init" in cfg_learner else "random"
    strategies = {}
    for i in game.set_bidder:
        strategies[i] = Strategy(i, game)
        strategies[i].initialize(init_method)

    # run soda
    return learner.run(mechanism, game, strategies)


def run_experiment(
    learn_alg: str,
    setting: str,
    experiment: str,
    logging: bool,
    save_strat: bool,
    num_runs: int,
    path: str,
    path_config: str,
):
    """Run Experiment

    Args:
        learn_alg (str): learning algorithm (soda, poga, frank_wolfe, ...)
        setting (str): mechanism
        experiment (str): specific experiment within defined setting/mechanism
        logging (bool): save logging file
        save_strat (bool): save strategies
        num_runs (int): number of repetitions for each experiment
        path (str): path where stuff is saved (rel. to directory where script is run)
        path_config (str): path to config file (rel. to soda-directory)
    """
    # directory to store results
    Path(path + "strategies/" + setting).mkdir(parents=True, exist_ok=True)

    # get parameter
    cfg, cfg_learner = get_config(path_config, setting, experiment, learn_alg)

    # initialize setting and compute utility
    t0 = time()
    mechanism, game = create_setting(setting, cfg)
    if not mechanism.own_gradient:
        print('Computations of Utilities for experiment: "' + experiment + '" started!')
        game.get_utility(mechanism)
    time_init = time() - t0

    # run soda
    for run in tqdm(
        range(num_runs),
        unit_scale=True,
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}",
    ):
        t0 = time()
        strategies = learn_strategies(mechanism, game, cfg_learner)
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
                game,
                learn_alg,
                experiment,
                setting,
                run,
                path,
            )

        if save_strat:
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

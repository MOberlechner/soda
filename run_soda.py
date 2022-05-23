from time import time

import hydra

from src.learner.soda import SODA
from src.util.logging import log_run
from src.util.setting import create_setting


def run_soda(mechanism, game, strategies, cfg_learner) -> None:
    """Runs SODA given the specified setting.

    Args:
        mechanism: continuous auction game
        game: approximation game
        strategies: dict of strategies
        cfg_soda: parameter for SODA
    """

    # init learner
    soda = SODA(
        cfg_learner.max_iter,
        cfg_learner.tol,
        cfg_learner.steprule_bool,
        cfg_learner.eta,
        cfg_learner.beta,
    )

    # initialize strategies
    for i in game.set_bidder:
        strategies[i].initialize("random")

    # run soda
    soda.run(mechanism, game, strategies)

    return strategies


if __name__ == "__main__":

    setting = "single_item"
    experiments_list = ["affiliated_values"]

    path = "experiment/" + setting + "/"
    hydra.initialize(config_path="configs/" + setting, job_name="run")
    logging = False
    runs = 1

    for experiment in experiments_list:

        # get parameter
        cfg_learner = hydra.compose(config_name="learner")
        cfg = hydra.compose(config_name=experiment)

        # initialize setting and compute utility
        t0 = time()
        mechanism, game, strategies = create_setting(setting, cfg)
        if not mechanism.own_gradient:
            print(
                'Computations of Utilities for experiment: "'
                + experiment
                + '" started!'
            )
            game.get_utility(mechanism)
        time_init = time() - t0

        # run soda
        for r in range(runs):
            t0 = time()
            strategies = run_soda(mechanism, game, strategies, cfg_learner)
            time_run = time() - t0

            # log and save
            if logging is True:
                log_run(strategies, experiment, r, time_init, time_run, path)

                for i in game.set_bidder:
                    name = experiment + ("_run_" + str(r) if runs > 1 else "")
                    strategies[i].save(name, path)

        print('Experiment: "' + experiment + '" finished!')

from time import time

import hydra

from src.learner import SODA
from src.util.logging import log_run
from src.util.setting import create_setting


def run_soda(mechanism, game, strategies):

    # parameter learner
    max_iter = 3000
    tol = 1e-4
    steprule_bool = True
    eta = 10
    beta = 1 / 20

    # compute utilities
    if not mechanism.own_gradient:
        game.get_utility(mechanism)

    # create learner
    soda = SODA(max_iter, tol, steprule_bool, eta, beta)

    # initialize strategies
    for i in game.set_bidder:
        strategies[i].initialize("random")

    # run soda
    soda.run(mechanism, game, strategies)

    return strategies


if __name__ == "__main__":

    setting = "contest_game"
    experiments_list = [
        "tullock_contest_3_val_0.5",
        "tullock_contest_3_val_1",
        "tullock_contest_3_val_2",
        "tullock_contest_3_val_5",
    ]

    path = "experiment/" + setting + "/"
    hydra.initialize(config_path="configs/" + setting, job_name="run")
    logging = True
    runs = 10

    for experiment in experiments_list:

        # get setting
        cfg = hydra.compose(config_name=experiment)

        # create setting
        t0 = time()
        mechanism, game, strategies = create_setting(setting, cfg)
        time_init = time() - t0

        # run soda
        for r in range(runs):
            t0 = time()
            strategies = run_soda(mechanism, game, strategies)
            time_run = time() - t0

            # log
            if logging is True:
                log_run(strategies, experiment, r, time_init, time_run, path)

            # save strategies
            for i in game.set_bidder:
                name = experiment + ("_run_" + str(r) if runs > 1 else "")
                strategies[i].save(name, path)

        print('Experiment: "' + experiment + '" finished!')

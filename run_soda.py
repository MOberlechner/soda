from time import time

import hydra

from src.learner import SODA
from src.util import create_setting, log_run


def run_soda(mechanism, game, strategies):

    # parameter learner
    max_iter = int(1e4)
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

    setting = "crowdsourcing"
    experiments_list = [
        "crowdsourcing_val2_no_ties_3_price1",
        "crowdsourcing_val2_no_ties_3_price2",
        "crowdsourcing_val2_no_ties_3_price3",
    ]

    path = "experiment/" + setting + "/"
    hydra.initialize(config_path="configs/" + setting, job_name="run")
    logging = True
    runs = 1

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

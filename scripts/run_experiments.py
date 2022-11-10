from pathlib import Path
from time import time

from tqdm import tqdm

from src.strategy import Strategy
from src.util.logging import log_sim, log_strat
from src.util.metrics import compute_l2_norm, compute_util_loss_scaled, compute_utility
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
    learner.run(mechanism, game, strategies)
    return strategies


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

    print('Experiment: "' + experiment + '" started!')
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


def run_sim(
    learn_alg,
    setting,
    experiment,
    path,
    path_config,
    num_runs: int = 1,
    n_obs: int = int(2 * 22),
    logging: bool = True,
    n_scaled: int = 1024,
    m_scaled: int = 1024,
):

    # get parameter
    cfg, cfg_learner = get_config(path_config, setting, experiment, learn_alg)

    # create settings (standard or scaled)
    if cfg.bne_known:
        mechanism, game = create_setting(setting, cfg)
    else:
        cfg.n = n_scaled
        cfg.m = m_scaled
        mechanism, game = create_setting(setting, cfg)

        if not mechanism.own_gradient:
            game.get_utility(mechanism)
            print("Utilties for experiments computed!")

    for run in range(num_runs):

        # import strategies: naming convention now includes learner !!!
        name = (
            cfg_learner.name
            + "_"
            + experiment
            + ("_run_" + str(run) if num_runs > 1 else "")
        )

        strategies = {}
        for i in game.set_bidder:
            strategies[i] = Strategy(i, game)

        for i in strategies:
            if cfg.bne_known:
                strategies[i].load(name, setting, path)
            else:
                strategies[i].load_scale(name, setting, path, n_scaled, m_scaled)

        # compute metrics if BNE is known
        if cfg.bne_known:
            l2_norm = compute_l2_norm(mechanism, strategies, n_obs)
            util_bne, util_vs_bne, util_loss = compute_utility(
                mechanism, strategies, n_obs
            )

            if logging is True:
                tag_labels = ["l2_norm", "util_in_bne", "util_vs_bne", "util_loss"]
                values = [l2_norm, util_bne, util_vs_bne, util_loss]
                for tag, val in zip(tag_labels, values):
                    log_sim(strategies, experiment, setting, run, tag, val, path)
        else:
            util_loss_approx = compute_util_loss_scaled(mechanism, game, strategies)

            if logging is True:
                log_sim(
                    strategies,
                    experiment,
                    run,
                    "util_loss_approx",
                    util_loss_approx,
                    path,
                )
    print('Simulation for experiments: "' + experiment + '" finished!')

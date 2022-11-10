import hydra

from src.strategy import Strategy
from src.util.logging import log_sim
from src.util.metrics import compute_l2_norm, compute_util_loss_scaled, compute_utility
from src.util.setup import create_learner, create_setting, get_config


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
                    r,
                    "util_loss_approx",
                    util_loss_approx,
                    path,
                )
    print('Simulation for experiments: "' + experiment + '" finished!')


# -------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":

    learn_alg = "frank_wolfe"
    setting = "single_item"
    experiments_list = [
        "fpsb",
    ]

    # specify path for experiments
    path = "experiment/"
    path_config = "configs/"
    logging = True
    num_runs = 1
    n_obs = int(2**22)
    n_scaled = m_scaled = 1024

    for experiment in experiments_list:
        run_sim(
            learn_alg,
            setting,
            experiment,
            path,
            path_config,
            num_runs,
            n_obs,
            logging,
            n_scaled,
            m_scaled,
        )

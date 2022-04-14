import hydra

from src.util.logging import log_sim
from src.util.metrics import compute_l2_norm, compute_util_loss_scaled, compute_utility
from src.util.setting import create_setting


def run_sim(
    setting,
    experiment,
    runs: int = 1,
    n_obs: int = int(2 * 22),
    logging: bool = True,
    factor: int = 8,
):

    # get setting
    cfg = hydra.compose(config_name=experiment)
    mechanism, game, strategies = create_setting(setting, cfg)

    # path to strategies
    path_dir = "experiment/" + setting + "/"

    # create scaled game for util_loss_large
    if not cfg.bne_known:
        cfg.n = factor * cfg.n
        cfg.m = factor * cfg.m
        mechanism_scaled, game_scaled, strategies_scaled = create_setting(setting, cfg)
        if not mechanism.own_gradient:
            game_scaled.get_utility(mechanism_scaled)

    for r in range(runs):

        # import strategies
        name = experiment + ("_run_" + str(r) if runs > 1 else "")
        for i in strategies:
            if cfg.bne_known:
                strategies[i].load(name, path_dir)
            else:
                strategies_scaled[i].load_scale(name, path_dir, factor)

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
                    log_sim(strategies, experiment, r, tag, val, path)
        else:
            util_loss_approx = compute_util_loss_scaled(
                mechanism_scaled, game_scaled, strategies_scaled
            )

            if logging is True:
                log_sim(
                    strategies,
                    experiment,
                    r,
                    "util_loss_approx",
                    util_loss_approx,
                    path,
                )


if __name__ == "__main__":

    setting = "contest_game"
    experiments_list = [
        "tullock_contest_2_val_0.5",
        "tullock_contest_2_val_1",
        "tullock_contest_2_val_2",
        "tullock_contest_2_val_5",
        "tullock_contest_3_val_0.5",
        "tullock_contest_3_val_1",
        "tullock_contest_3_val_2",
        "tullock_contest_3_val_5",
    ]

    path = "experiment/" + setting + "/"
    hydra.initialize(config_path="configs/" + setting, job_name="run")
    logging = True
    runs = 10
    n_obs = int(2 ** 22)
    bne_known = True
    factor = 8

    for experiment in experiments_list:

        # run simulation
        run_sim(setting, experiment, runs, n_obs, logging, bne_known, factor)

        print('Simulation for experiments: "' + experiment + '" finished!')

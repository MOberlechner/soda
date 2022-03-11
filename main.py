import hydra

from src.game import Game
from src.learner import SODA
from src.mechanism.contest_game import ContestGame
from src.strategy import Strategy


def create_setting(setting: str, cfg):

    if setting == "contest_game":
        mechanism = ContestGame(
            cfg.bidder,
            cfg.o_space,
            cfg.a_space,
            cfg.param_prior,
            cfg.csf,
            cfg.param_csf,
        )
    else:
        raise ValueError('Mechanism "' + setting + '" not available')

    # create approximation game
    game = Game(mechanism, cfg.n, cfg.m)

    # create and initialize strategies
    strategies = {}
    for i in game.set_bidder:
        strategies[i] = Strategy(i, game)

    return mechanism, game, strategies


def run_soda(mechanism, game, strategies):

    # parameter learner
    max_iter = int(2e3)
    tol = 1e-4
    steprule_bool = True
    eta = 10
    beta = 1 / 20

    # compute utilities
    game.get_utility(mechanism)

    # create learner
    soda = SODA(max_iter, tol, steprule_bool, eta, beta)

    # initialize strategies
    for i in game.set_bidder:
        strategies[i].initialize("random")

    # run soda
    soda.run(game, strategies)

    return strategies


def get_config(setting: str, experiment: str):
    hydra.initialize(config_path="configs/" + setting, job_name="run")
    cfg = hydra.compose(config_name=experiment)
    return cfg


if __name__ == "__main__":

    setting = "contest_game"
    experiment = "tullock_contest_convex1"

    path = "experiment/" + setting + "/"
    runs = 1

    # create setting
    cfg = get_config(setting, experiment)
    mechanism, game, strategies = create_setting(setting, cfg)
    # run soda
    for r in range(runs):
        strategies = run_soda(mechanism, game, strategies)
        # save strategies
        for i in game.set_bidder:
            name = experiment + ("_run_" + str(r) if runs > 1 else "")
            strategies[i].save(name, path)

    print("Done!")

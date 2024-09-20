from itertools import product
from time import time

from projects.ad_auctions.config_exp import (
    LOGGING,
    NUMBER_SAMPLES,
    PATH_TO_CONFIGS,
    PATH_TO_EXPERIMENTS,
    ROUND_DECIMALS,
    SAVE_STRAT,
)
from soda.util.experiment import Experiment

NUMBER_RUNS = 10
LEARNING = True
SIMULATION = True

EXPERIMENT_TAG = "revenue"

games_uniform = [
    f"revenue/{util_type}_{payment_rule}_{n_bidder}.yaml"
    for n_bidder in [2, 3, 5, 10]
    for payment_rule in ["fp", "sp"]
    for util_type in ["ql", "roi", "rosb"]
]
games_gaussian = [
    f"revenue/gaus_{util_type}_{payment_rule}_{n_bidder}.yaml"
    for n_bidder in [2, 3, 5, 10]
    for payment_rule in ["fp", "sp"]
    for util_type in ["ql", "roi", "rosb"]
]
learner = [
    "soda1.yaml",
]
experiment_list = list(product(games_uniform + games_gaussian, learner))

if __name__ == "__main__":
    print(f"\nRunning {len(experiment_list)} Experiments".ljust(100, "."), "\n")
    t0 = time()
    successfull = 0
    for config_game, config_learner in experiment_list:

        exp_handler = Experiment(
            PATH_TO_CONFIGS + "game/" + config_game,
            PATH_TO_CONFIGS + "learner/" + config_learner,
            NUMBER_RUNS,
            LEARNING,
            SIMULATION,
            LOGGING,
            SAVE_STRAT,
            NUMBER_SAMPLES,
            PATH_TO_EXPERIMENTS,
            ROUND_DECIMALS,
            EXPERIMENT_TAG,
        )
        exp_handler.run()
        successfull += 1 - exp_handler.error
    t1 = time()
    print(
        f"\n{successfull} out of {len(experiment_list)} experiments successfull ({(t1-t0)/60:.1f} min)".ljust(
            100, "."
        ),
        "\n",
    )

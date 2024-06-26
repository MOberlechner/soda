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

NUMBER_RUNS = 1
LEARNING = True
SIMULATION = False

EXPERIMENT_TAG = "baseline"
games = [
    "baseline/ql_fp_3.yaml",
    "baseline/ql_sp_3.yaml",
    "baseline/roi_fp_3.yaml",
    "baseline/roi_sp_3.yaml",
    "baseline/rosb_fp_3.yaml",
    "baseline/rosb_sp_3.yaml",
]

learner = [
    "soma2_baseline.yaml",
]

experiment_list = list(product(games, learner))

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

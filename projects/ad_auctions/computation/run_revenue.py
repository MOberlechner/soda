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
    "revenue/ql_fp_2.yaml",
    "revenue/ql_sp_2.yaml",
    "revenue/roi_fp_2.yaml",
    "revenue/roi_sp_2.yaml",
    "revenue/rosb_fp_2.yaml",
    "revenue/rosb_sp_2.yaml",
    "revenue/ql_fp_3.yaml",
    "revenue/ql_sp_3.yaml",
    "revenue/roi_fp_3.yaml",
    "revenue/roi_sp_3.yaml",
]

games_gaussian = [
    "revenue/gaus_ql_fp_2.yaml",
    "revenue/gaus_ql_sp_2.yaml",
    "revenue/gaus_roi_fp_2.yaml",
    "revenue/gaus_roi_sp_2.yaml",
    "revenue/gaus_rosb_fp_2.yaml",
    "revenue/gaus_rosb_sp_2.yaml",
]

learner = [
    "soda1_revenue.yaml",
]

experiment_list = list(product(games_uniform + games_gaussian, learner))

if __name__ == "__main__":
    print(f"\nRunning {len(experiment_list)} Experiments".ljust(100, "."), "\n")
    t0 = time()

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
    t1 = time()
    print(f"\nExperiments finished ({(t1-t0)/60:.1f} min)".ljust(100, "."), "\n")

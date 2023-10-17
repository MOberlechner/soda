from itertools import product
from time import time

from soda.util.experiment import Experiment

PATH_TO_CONFIGS = "projects/ad-auctions/configs/"
PATH_TO_EXPERIMENTS = "experiments/ad_auctions/"

NUMBER_RUNS = 1
LEARNING = (True,)
SIMULATION = (False,)

LOGGING = True
SAVE_STRAT = True
NUMBER_SAMPLES = int(2**22)
ROUND_DECIMALS = 3

EXPERIMENT_TAG = "baseline"
games = [
    "baseline_ql_fp_3.yaml",
    "baseline_ql_sp_3.yaml",
    "baseline_roi_fp_3.yaml",
    "baseline_roi_sp_3.yaml",
    "baseline_rosb_fp_3.yaml",
    "baseline_rosb_sp_3.yaml",
]

learner = [
    "soda2_baseline.yaml",
]

experiment_list = list(product(games, learner))

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

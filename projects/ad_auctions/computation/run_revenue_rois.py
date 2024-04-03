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

EXPERIMENT_TAG = "revenue_rois"
# {rois_parameter}_{payment_rule}_{n_agents}_{budget}
games_uniform = [
    # fix budget up to 1.0
    "revenue_rois/rois1_fp_2_b1.yaml",
    "revenue_rois/rois2_fp_2_b1.yaml",
    "revenue_rois/rois3_fp_2_b1.yaml",
    "revenue_rois/rois4_fp_2_b1.yaml",
    "revenue_rois/rois5_fp_2_b1.yaml",
    "revenue_rois/rois1_sp_2_b1.yaml",
    "revenue_rois/rois2_sp_2_b1.yaml",
    "revenue_rois/rois3_sp_2_b1.yaml",
    "revenue_rois/rois4_sp_2_b1.yaml",
    "revenue_rois/rois5_sp_2_b1.yaml",
    # fix budget up to 0.6
    "revenue_rois/rois1_fp_2_b2.yaml",
    "revenue_rois/rois2_fp_2_b2.yaml",
    "revenue_rois/rois3_fp_2_b2.yaml",
    "revenue_rois/rois4_fp_2_b2.yaml",
    "revenue_rois/rois5_fp_2_b2.yaml",
    "revenue_rois/rois1_sp_2_b2.yaml",
    "revenue_rois/rois2_sp_2_b2.yaml",
    "revenue_rois/rois3_sp_2_b2.yaml",
    "revenue_rois/rois4_sp_2_b2.yaml",
    "revenue_rois/rois5_sp_2_b2.yaml",
]

learner = [
    "soda1_revenue.yaml",
]

experiment_list = list(product(games_uniform, learner))

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

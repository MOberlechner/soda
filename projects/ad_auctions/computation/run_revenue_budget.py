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

EXPERIMENT_TAG = "revenue_budget"
# {rois_parameter}_{payment_rule}_{n_agents}_{budget}
games_uniform = [
    # buget (1.01) with parameter 1.0
    "revenue_budget/ql_fp_2_b1.yaml",
    "revenue_budget/roi_fp_2_b1.yaml",
    "revenue_budget/ros_fp_2_b1.yaml",
    "revenue_budget/ql_sp_2_b1.yaml",
    "revenue_budget/roi_sp_2_b1.yaml",
    "revenue_budget/ros_sp_2_b1.yaml",
    "revenue_budget/ql_fp_3_b1.yaml",
    "revenue_budget/roi_fp_3_b1.yaml",
    "revenue_budget/ros_fp_3_b1.yaml",
    "revenue_budget/ql_sp_3_b1.yaml",
    "revenue_budget/roi_sp_3_b1.yaml",
    "revenue_budget/ros_sp_3_b1.yaml",
    # buget (1.01) with parameter 0.5
    "revenue_budget/ql_fp_2_b2.yaml",
    "revenue_budget/roi_fp_2_b2.yaml",
    "revenue_budget/ros_fp_2_b2.yaml",
    "revenue_budget/ql_sp_2_b2.yaml",
    "revenue_budget/roi_sp_2_b2.yaml",
    "revenue_budget/ros_sp_2_b2.yaml",
    "revenue_budget/ql_fp_3_b2.yaml",
    "revenue_budget/roi_fp_3_b2.yaml",
    "revenue_budget/ros_fp_3_b2.yaml",
    "revenue_budget/ql_sp_3_b2.yaml",
    "revenue_budget/roi_sp_3_b2.yaml",
    "revenue_budget/ros_sp_3_b2.yaml",
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

from itertools import product
from time import time

from projects.ad_auctions.config_exp import *
from soda.util.experiment import run_experiments

LABEL_EXPERIMENT = "revenue_rois"
games_uniform = [
    # first price
    "revenue_rois/rois1_fp_2.yaml",
    "revenue_rois/rois2_fp_2.yaml",
    "revenue_rois/rois3_fp_2.yaml",
    "revenue_rois/rois4_fp_2.yaml",
    "revenue_rois/rois5_fp_2.yaml",
    # second price
    "revenue_rois/rois1_sp_2.yaml",
    "revenue_rois/rois2_sp_2.yaml",
    "revenue_rois/rois3_sp_2.yaml",
    "revenue_rois/rois4_sp_2.yaml",
    "revenue_rois/rois5_sp_2.yaml",
]
learner = [
    "soda1_revenue.yaml",
]

experiment_list = list(product(games_uniform, learner))

if __name__ == "__main__":
    run_experiments(
        experiment_list,
        path_to_configs=PATH_TO_CONFIGS,
        number_runs=NUMBER_RUNS,
        label_experiment=LABEL_EXPERIMENT,
        param_computation=PARAM_COMPUTATION,
        param_simulation=PARAM_SIMULATION,
        param_evaluation={"active": False},
        param_logging=PARAM_LOGGING,
    )

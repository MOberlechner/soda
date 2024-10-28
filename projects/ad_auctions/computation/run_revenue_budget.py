import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from itertools import product
from time import time

from projects.ad_auctions.config_exp import *
from soda.util.experiment import run_experiments

LABEL_EXPERIMENT = "revenue_budget"
games = [
    # buget (1.01)
    "revenue_budget/ql_fp_2_b1.yaml",
    "revenue_budget/roi_fp_2_b1.yaml",
    "revenue_budget/ros_fp_2_b1.yaml",
    "revenue_budget/ql_sp_2_b1.yaml",
    "revenue_budget/roi_sp_2_b1.yaml",
    "revenue_budget/ros_sp_2_b1.yaml",
    # buget (0.81)
    "revenue_budget/ql_fp_2_b2.yaml",
    "revenue_budget/roi_fp_2_b2.yaml",
    "revenue_budget/ros_fp_2_b2.yaml",
    "revenue_budget/ql_sp_2_b2.yaml",
    "revenue_budget/roi_sp_2_b2.yaml",
    "revenue_budget/ros_sp_2_b2.yaml",
]
learner = [
    "soda1.yaml",
]

experiment_list = list(product(games, learner))

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

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from itertools import product
from time import time

from projects.ad_auctions.config_exp import *
from soda.util.experiment import run_experiments

LABEL_EXPERIMENT = "revenue"

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

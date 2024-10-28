from itertools import combinations_with_replacement, product
from time import time

from projects.ad_auctions.config_exp import *
from soda.util.experiment import run_experiments

LABEL_EXPERIMENT = "revenue_asym"

games_uniform = [
    f"revenue_asym/{payment_rule}2_{util1}_{util2}.yaml"
    for payment_rule in ["fp", "sp"]
    for util1, util2 in (combinations_with_replacement(["ql", "roi", "ros"], 2))
]
games_gaussian = [
    f"revenue_asym/{payment_rule}2_{util1}_{util2}_gaus.yaml"
    for payment_rule in ["fp", "sp"]
    for util1, util2 in (combinations_with_replacement(["ql", "roi", "ros"], 2))
]
learner = [
    "sofw.yaml",
]

experiment_list = list(product(games_gaussian, learner))

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

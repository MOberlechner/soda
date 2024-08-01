from itertools import product
from time import time

from projects.contests.config_exp import *
from soda.util.experiment import run_experiments

label_experiment = "tullock"
game_contests = [
    # symmetric, different csf parameters
    "tullock/bidder2_csf1.yaml",
    "tullock/bidder2_csf2.yaml",
    "tullock/bidder2_csf3.yaml",
    "tullock/bidder2_csf4.yaml",
    # asymmetric, different strong agents
    "tullock/bidder2_asym1.yaml",
    "tullock/bidder2_asym2.yaml",
    "tullock/bidder2_asym3.yaml",
    "tullock/bidder2_asym4.yaml",
]
learner_contests = [
    "soda2_eta10_beta05.yaml",
]
experiment_list = list(product(game_contests, learner_contests))

if __name__ == "__main__":
    run_experiments(
        experiment_list,
        path_to_configs=PATH_TO_CONFIGS,
        number_runs=NUMBER_RUNS,
        label_experiment=label_experiment,
        param_computation=PARAM_COMPUTATION,
        param_simulation={"active": False},
        param_evaluation={"active": False},
        param_logging=PARAM_LOGGING,
    )

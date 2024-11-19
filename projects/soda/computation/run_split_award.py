import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from itertools import product
from time import time

from projects.soda.config_exp import *
from soda.util.experiment import run_experiments

LABEL_EXPERIMENT = "split_award"
game_gaussian = ["split_award/sa_gaussian.yaml"]
game_uniform = ["split_award/sa_uniform.yaml"]
learner_gaussian = [
    "soda1_eta20_beta05.yaml",
    "soda2_eta005_beta05.yaml",
    "soma2_eta005_beta50.yaml",
    "sofw.yaml",
    "fp.yaml",
]
learner_uniform = [
    "soda1_eta20_beta05.yaml",
    "soda2_eta005_beta05.yaml",
    "soma2_eta001_beta50.yaml",
    "sofw.yaml",
    "fp.yaml",
]

experiment_list = list(product(game_gaussian, learner_gaussian)) + list(
    product(game_uniform, learner_uniform)
)

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

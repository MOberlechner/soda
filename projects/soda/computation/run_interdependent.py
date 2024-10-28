import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from itertools import product
from time import time

from projects.soda.config_exp import *
from soda.util.experiment import run_experiments

LABEL_EXPERIMENT = "interdependent"
game_affiliated = [
    "interdependent/affiliated_values.yaml",
]
game_common = [
    "interdependent/common_value.yaml",
]
learner_affiliated = [
    "soda1_eta100_beta50.yaml",
    "soda2_eta1_beta50.yaml",
    "soma2_eta1_beta50.yaml",
    "sofw.yaml",
    "fp.yaml",
]
learner_common = [
    "soda1_eta100_beta50.yaml",
    "soda2_eta1_beta05.yaml",
    "soma2_eta50_beta50.yaml",
    "sofw.yaml",
    "fp.yaml",
]
experiment_list = list(product(game_affiliated, learner_affiliated)) + list(
    product(game_common, learner_common)
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

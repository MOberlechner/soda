import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from itertools import product
from time import time

from projects.soda.config_exp import *
from soda.util.experiment import run_experiments

LABEL_EXPERIMENT = "discretization"
game_discr_fast = [
    "discretization/fast_2_discr016.yaml",
    "discretization/fast_2_discr032.yaml",
    "discretization/fast_2_discr064.yaml",
    "discretization/fast_2_discr128.yaml",
    "discretization/fast_2_discr256.yaml",
    "discretization/fast_10_discr016.yaml",
    "discretization/fast_10_discr032.yaml",
    "discretization/fast_10_discr064.yaml",
    "discretization/fast_10_discr128.yaml",
]
game_discr_gen = [
    "discretization/general_2_discr016.yaml",
    "discretization/general_2_discr032.yaml",
    "discretization/general_2_discr064.yaml",
    "discretization/general_2_discr128.yaml",
    "discretization/general_3_discr016.yaml",
    "discretization/general_3_discr032.yaml",
    "discretization/general_3_discr064.yaml",
    "discretization/general_3_discr128.yaml",
]
learner = [
    "soda1_eta10_beta05.yaml",
]

experiment_list_fast = list(product(game_discr_fast, learner))
experiment_list_gen = list(product(game_discr_gen, learner))

if __name__ == "__main__":
    run_experiments(
        experiment_list_fast,
        path_to_configs=PATH_TO_CONFIGS,
        number_runs=NUMBER_RUNS,
        label_experiment=LABEL_EXPERIMENT,
        param_computation=PARAM_COMPUTATION,
        param_simulation={"active": False},
        param_evaluation={"active": False},
        param_logging=PARAM_LOGGING,
    )

    run_experiments(
        experiment_list_gen,
        path_to_configs=PATH_TO_CONFIGS,
        number_runs=NUMBER_RUNS,
        label_experiment=LABEL_EXPERIMENT,
        param_computation=PARAM_COMPUTATION,
        param_simulation=PARAM_SIMULATION,
        param_evaluation={"active": False},
        param_logging=PARAM_LOGGING,
    )

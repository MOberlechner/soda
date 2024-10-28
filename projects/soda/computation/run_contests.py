import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from itertools import product
from time import time

from projects.soda.config_exp import *
from soda.util.experiment import run_experiments

LABEL_EXPERIMENT = "contests"
game_contests = [
    "contests/asym_ratio1.yaml",
    "contests/asym_ratio2.yaml",
    "contests/asym_ratio3.yaml",
    "contests/sym_ratio1.yaml",
    "contests/sym_ratio2.yaml",
    "contests/sym_ratio3.yaml",
]
learner_contests = [
    "soda1_eta100_beta05.yaml",
    "soda2_eta10_beta05.yaml",
    "soma2_eta100_beta50.yaml",
    "sofw.yaml",
    "fp.yaml",
]
experiment_list = list(product(game_contests, learner_contests))

if __name__ == "__main__":
    run_experiments(
        experiment_list,
        path_to_configs=PATH_TO_CONFIGS,
        number_runs=1,
        label_experiment=LABEL_EXPERIMENT,
        param_computation=PARAM_COMPUTATION,
        param_simulation={"active": False},
        param_evaluation={"active": False},
        param_logging=PARAM_LOGGING,
    )

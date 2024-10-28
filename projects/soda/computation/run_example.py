import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from itertools import product
from time import time

from projects.soda.config_exp import *
from soda.util.experiment import run_experiments

LABEL_EXPERIMENT = "example"
game = [
    "example/fpsb_2_discr020.yaml",
    # only used for visualization in appendix:
    # "example/fpsb_2_discr032.yaml",
]
learner = [
    "soda1_eta20_beta05.yaml",
]
experiment_list = list(product(game, learner))

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

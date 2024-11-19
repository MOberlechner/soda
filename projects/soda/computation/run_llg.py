import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from itertools import product
from time import time

from projects.soda.config_exp import *
from soda.util.experiment import run_experiments

LABEL_EXPERIMENT = "llg"
game_llg = [
    "llg/nb_gamma1.yaml",
    "llg/nb_gamma2.yaml",
    "llg/nb_gamma3.yaml",
    "llg/nz_gamma1.yaml",
    "llg/nz_gamma2.yaml",
    "llg/nz_gamma3.yaml",
    "llg/nvcg_gamma1.yaml",
    "llg/nvcg_gamma2.yaml",
    "llg/nvcg_gamma3.yaml",
]
game_llg_fp = [
    "llg/fp_gamma1.yaml",
    "llg/fp_gamma2.yaml",
    "llg/fp_gamma3.yaml",
]
learner_llg = [
    "soda1_eta100_beta05.yaml",
    "soda2_eta50_beta05.yaml",
    "soma2_eta50_beta05.yaml",
    "sofw.yaml",
    "fp.yaml",
]
learner_llg_fp = ["sofw.yaml"]

experiment_list = list(product(game_llg, learner_llg)) + list(
    product(game_llg_fp, learner_llg_fp)
)
experiment_list_fp = list(product(game_llg_fp, learner_llg_fp))

if __name__ == "__main__":

    run_experiments(
        experiment_list,
        path_to_configs=PATH_TO_CONFIGS,
        number_runs=10,
        label_experiment=LABEL_EXPERIMENT,
        param_computation=PARAM_COMPUTATION,
        param_simulation=PARAM_SIMULATION,
        param_evaluation={"active": False},
        param_logging=PARAM_LOGGING,
    )

    run_experiments(
        experiment_list_fp,
        path_to_configs=PATH_TO_CONFIGS,
        number_runs=1,
        label_experiment=LABEL_EXPERIMENT,
        param_computation=PARAM_COMPUTATION,
        param_simulation={"active": False},
        param_evaluation={"active": False},
        param_logging=PARAM_LOGGING,
    )

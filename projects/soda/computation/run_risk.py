import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from itertools import product
from time import time

from projects.soda.config_exp import *
from soda.util.experiment import run_experiments

LABEL_EXPERIMENT = "risk"
game_allpay = [
    "risk/allpay_risk0.yaml",
    "risk/allpay_risk1.yaml",
    "risk/allpay_risk2.yaml",
    "risk/allpay_risk3.yaml",
    "risk/allpay_risk4.yaml",
    "risk/allpay_risk5.yaml",
]
learner_allpay = [
    "soda1_eta25_beta05.yaml",
]
game_fpsb = [
    "risk/fpsb_risk0.yaml",
    "risk/fpsb_risk1.yaml",
    "risk/fpsb_risk2.yaml",
    "risk/fpsb_risk3.yaml",
    "risk/fpsb_risk4.yaml",
    "risk/fpsb_risk5.yaml",
]
learner_fpsb = [
    "soda1_eta20_beta05.yaml",
    "soda2_eta01_beta05.yaml",
    "soma2_eta05_beta50.yaml",
    "sofw.yaml",
    "fp.yaml",
]
experiment_list = list(product(game_allpay, learner_allpay)) + list(
    product(game_fpsb, learner_fpsb)
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

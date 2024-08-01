from itertools import product
from time import time

from projects.ad_auctions.config_exp import *
from soda.util.experiment import run_experiments

LABEL_EXPERIMENT = "baseline"
games = [
    "baseline/ql_fp_3.yaml",
    "baseline/ql_sp_3.yaml",
    "baseline/roi_fp_3.yaml",
    "baseline/roi_sp_3.yaml",
    "baseline/rosb_fp_3.yaml",
    "baseline/rosb_sp_3.yaml",
]
learner = [
    "soma2_baseline.yaml",
]
experiment_list = list(product(games, learner))

if __name__ == "__main__":
    run_experiments(
        experiment_list,
        path_to_configs=PATH_TO_CONFIGS,
        number_runs=1,
        label_experiment=LABEL_EXPERIMENT,
        param_computation={"active": True, "init_method": "equal"},
        param_simulation={"active": False},
        param_evaluation={"active": False},
        param_logging=PARAM_LOGGING,
    )

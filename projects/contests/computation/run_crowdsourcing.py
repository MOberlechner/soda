import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from itertools import product

from projects.contests.config_exp import *
from soda.util.experiment import run_experiments

label_experiment = "crowdsourcing"
game_contests = [
    # 3 bidders, 2 prizes
    "crowdsourcing/bidder3_price1.yaml",
    "crowdsourcing/bidder3_price2.yaml",
    "crowdsourcing/bidder3_price3.yaml",
    # 5 bidders, 2 prizes
    "crowdsourcing/bidder5_price1.yaml",
    "crowdsourcing/bidder5_price2.yaml",
    "crowdsourcing/bidder5_price3.yaml",
    # 10 bidders, 2 prizes
    "crowdsourcing/bidder10_price1.yaml",
    "crowdsourcing/bidder10_price2.yaml",
    "crowdsourcing/bidder10_price3.yaml",
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
        param_simulation=PARAM_SIMULATION,
        param_evaluation={"active": False},
        param_logging=PARAM_LOGGING,
    )

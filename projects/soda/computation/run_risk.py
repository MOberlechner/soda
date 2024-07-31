from itertools import product
from time import time

from projects.soda.config_exp import *
from soda.util.experiment import Experiment

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
    print(f"\nRunning {len(experiment_list)} Experiments".ljust(100, "."), "\n")
    t0 = time()
    successfull = 0
    for config_game, config_learner in experiment_list:
        exp_handler = Experiment(
            PATH_TO_CONFIGS + "game/" + config_game,
            PATH_TO_CONFIGS + "learner/" + config_learner,
            number_runs=NUMBER_RUNS,
            label_experiment="risk",
            param_computation=PARAM_COMPUTATION,
            param_simulation=PARAM_SIMULATION,
            param_logging=PARAM_LOGGING,
        )
        exp_handler.run()
        successfull += 1 - exp_handler.error
    t1 = time()
    print(
        f"\n{successfull} out of {len(experiment_list)} experiments successfull ({(t1-t0)/60:.1f} min)".ljust(
            100, "."
        ),
        "\n",
    )

from itertools import product
from time import time

from projects.soda.config_exp import *
from soda.util.experiment import Experiment

allpay_game = [
    "risk/allpay_risk0.yaml",
    "risk/allpay_risk1.yaml",
    "risk/allpay_risk2.yaml",
    "risk/allpay_risk3.yaml",
    "risk/allpay_risk4.yaml",
    "risk/allpay_risk5.yaml",
]
allpay_learner = [
    "soda1_eta25_beta05.yaml",
]
fpsb_game = [
    "risk/fpsb_risk0.yaml",
    "risk/fpsb_risk1.yaml",
    "risk/fpsb_risk2.yaml",
    "risk/fpsb_risk3.yaml",
    "risk/fpsb_risk4.yaml",
    "risk/fpsb_risk5.yaml",
]
fpsb_learner = [
    "soda1_eta20_beta05.yaml",
    "soda2_eta01_beta05.yaml",
    "soma2_eta05_beta50.yaml",
    "sofw.yaml",
    "fp.yaml",
]
experiment_list = list(product(allpay_game, allpay_learner)) + list(
    product(fpsb_game, fpsb_learner)
)

if __name__ == "__main__":
    print(f"\nRunning {len(experiment_list)} Experiments".ljust(100, "."), "\n")
    t0 = time()

    for config_game, config_learner in experiment_list:

        exp_handler = Experiment(
            PATH_TO_CONFIGS + "game/" + config_game,
            PATH_TO_CONFIGS + "learner/" + config_learner,
            NUMBER_RUNS,
            LEARNING,
            SIMULATION,
            LOGGING,
            SAVE_STRAT,
            NUMBER_SAMPLES,
            SAVE_INIT_STRAT,
            PATH_TO_EXPERIMENTS,
            ROUND_DECIMALS,
        )
        exp_handler.run()
    t1 = time()
    print(f"\nExperiments finished ({(t1-t0)/60:.1f} min)".ljust(100, "."), "\n")

from itertools import product
from time import time

from projects.soda.config_exp import *
from soda.util.experiment import Experiment

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

SIMULATION = False

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
            PATH_TO_EXPERIMENTS,
            ROUND_DECIMALS,
            experiment_tag="contests",
        )
        exp_handler.run()
    t1 = time()
    print(f"\nExperiments finished ({(t1-t0)/60:.1f} min)".ljust(100, "."), "\n")

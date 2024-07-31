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

if __name__ == "__main__":
    print(f"\nRunning {len(experiment_list)} Experiments".ljust(100, "."), "\n")
    t0 = time()
    successfull = 0
    for config_game, config_learner in experiment_list:
        exp_handler = Experiment(
            PATH_TO_CONFIGS + "game/" + config_game,
            PATH_TO_CONFIGS + "learner/" + config_learner,
            number_runs=1,
            label_experiment="contests",
            param_computation=PARAM_COMPUTATION,
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

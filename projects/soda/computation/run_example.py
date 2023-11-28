from itertools import product
from time import time

from projects.soda.config_exp import *
from soda.util.experiment import Experiment

game = [
    "example/fpsb_2_discr020.yaml",
    # only used for visualization in appendix:
    # "example/fpsb_2_discr032.yaml",
]
learner = [
    "soda1_eta20_beta05.yaml",
]
experiment_list = list(product(game, learner))

SIMULATION = False
NUMBER_RUNS = 1

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
            experiment_tag="example",
        )
        exp_handler.run()
    t1 = time()
    print(
        f"\n {len(experiment_list)} Experiments finished in {(t1-t0)/60:.1f} min".ljust(
            100, "."
        ),
        "\n",
    )

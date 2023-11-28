from itertools import product
from time import time

from projects.soda.config_exp import *
from soda.util.experiment import Experiment

game_discretization = [
    "discretization/fast_2_discr016.yaml",
    "discretization/fast_2_discr032.yaml",
    "discretization/fast_2_discr064.yaml",
    "discretization/fast_2_discr128.yaml",
    "discretization/fast_2_discr256.yaml",
    "discretization/fast_10_discr016.yaml",
    "discretization/fast_10_discr032.yaml",
    "discretization/fast_10_discr064.yaml",
    "discretization/fast_10_discr128.yaml",
    "discretization/general_2_discr016.yaml",
    "discretization/general_2_discr032.yaml",
    "discretization/general_2_discr064.yaml",
    "discretization/general_2_discr128.yaml",
    "discretization/general_3_discr016.yaml",
    "discretization/general_3_discr032.yaml",
    "discretization/general_3_discr064.yaml",
    "discretization/general_3_discr128.yaml",
]
learner = [
    "soda1_eta10_beta05.yaml",
]

experiment_list = list(product(game_discretization, learner))

if __name__ == "__main__":

    print(f"\nRunning {len(experiment_list)} Experiments".ljust(100, "."), "\n")
    t0 = time()

    for config_game, config_learner in experiment_list:

        SIMULATION = True if "fast_2_" in config_game else False

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
            experiment_tag="discretization",
        )
        exp_handler.run()
    t1 = time()
    print(
        f"\n {len(experiment_list)} Experiments finished in {(t1-t0)/60:.1f} min".ljust(
            100, "."
        ),
        "\n",
    )

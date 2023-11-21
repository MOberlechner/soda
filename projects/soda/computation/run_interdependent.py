from itertools import product
from time import time

from projects.soda.config_exp import *
from soda.util.experiment import Experiment

game_affiliated = [
    "interdependent/affiliated_values.yaml",
]
game_common = [
    "interdependent/common_value.yaml",
]
learner_affiliated = [
    "soda1_eta100_beta50.yaml",
    "soda2_eta1_beta50.yaml",
    "soma2_eta1_beta50.yaml",
    "sofw.yaml",
    "fp.yaml",
]
learner_common = [
    "soda1_eta100_beta50.yaml",
    "soda2_eta1_beta05.yaml",
    "soma2_eta50_beta50.yaml",
    "sofw.yaml",
    "fp.yaml",
]
experiment_list = list(product(game_affiliated, learner_affiliated)) + list(
    product(game_common, learner_common)
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
            PATH_TO_EXPERIMENTS,
            ROUND_DECIMALS,
            experiment_tag="interdependent",
        )
        exp_handler.run()
    t1 = time()
    print(
        f"\n {len(experiment_list)} Experiments finished in {(t1-t0)/60:.1f} min".ljust(
            100, "."
        ),
        "\n",
    )

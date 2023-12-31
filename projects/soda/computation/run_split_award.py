from itertools import product
from time import time

from projects.soda.config_exp import *
from soda.util.experiment import Experiment

game_gaussian = ["split_award/sa_gaussian.yaml"]
game_uniform = ["split_award/sa_uniform.yaml"]

learner_gaussian = [
    "soda1_eta20_beta05.yaml",
    "soda2_eta005_beta05.yaml",
    "soma2_eta005_beta50.yaml",
    "sofw.yaml",
    "fp.yaml",
]
learner_uniform = [
    "soda1_eta20_beta05.yaml",
    "soda2_eta005_beta05.yaml",
    "soma2_eta001_beta50.yaml",
    "sofw.yaml",
    "fp.yaml",
]

experiment_list = list(product(game_gaussian, learner_gaussian)) + list(
    product(game_uniform, learner_uniform)
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
            experiment_tag="split_award",
        )
        exp_handler.run()
    t1 = time()
    print(
        f"\n {len(experiment_list)} Experiments finished in {(t1-t0)/60:.1f} min".ljust(
            100, "."
        ),
        "\n",
    )

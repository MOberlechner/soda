from itertools import product
from time import time

from projects.soda.config_exp import *
from soda.util.experiment import Experiment

game_llg = [
    "llg/nb_gamma1.yaml",
    "llg/nb_gamma2.yaml",
    "llg/nb_gamma3.yaml",
    "llg/nz_gamma1.yaml",
    "llg/nz_gamma2.yaml",
    "llg/nz_gamma3.yaml",
    "llg/nvcg_gamma1.yaml",
    "llg/nvcg_gamma2.yaml",
    "llg/nvcg_gamma3.yaml",
]
game_llg_fp = [
    "llg/fp_gamma1.yaml",
    "llg/fp_gamma2.yaml",
    "llg/fp_gamma3.yaml",
]
learner_llg = [
    "soda1_eta100_beta05.yaml",
    "soda2_eta50_beta05.yaml",
    "soma2_eta50_beta05.yaml",
    "sofw.yaml",
    "fp.yaml",
]
learner_llg_fp = ["sofw.yaml"]

experiment_list = list(product(game_llg, learner_llg))
experiment_list_fp = list(product(game_llg_fp, learner_llg_fp))

if __name__ == "__main__":
    print(
        f"\nRunning {len(experiment_list) + len(experiment_list_fp)} Experiments".ljust(
            100, "."
        ),
        "\n",
    )
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
            experiment_tag="llg",
        )
        exp_handler.run()

    # No BNE for first-price setting, i.e., no simulation and 1 run
    NUMBER_RUNS = 1
    SIMULATION = False

    for config_game, config_learner in experiment_list_fp:
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
            experiment_tag="llg",
        )
        exp_handler.run()

    t1 = time()
    print(
        f"\n {len(experiment_list)} Experiments finished in {(t1-t0)/60:.1f} min".ljust(
            100, "."
        ),
        "\n",
    )

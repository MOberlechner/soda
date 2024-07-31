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
    successfull = 0
    for config_game, config_learner in experiment_list:
        exp_handler = Experiment(
            PATH_TO_CONFIGS + "game/" + config_game,
            PATH_TO_CONFIGS + "learner/" + config_learner,
            number_runs=NUMBER_RUNS,
            label_experiment="llg",
            param_computation=PARAM_COMPUTATION,
            param_simulation=PARAM_SIMULATION,
            param_logging=PARAM_LOGGING,
        )
        exp_handler.run()
        successfull += 1 - exp_handler.error

    for config_game, config_learner in experiment_list_fp:
        exp_handler = Experiment(
            PATH_TO_CONFIGS + "game/" + config_game,
            PATH_TO_CONFIGS + "learner/" + config_learner,
            number_runs=1,
            label_experiment="llg",
            param_computation=PARAM_COMPUTATION,
            param_logging=PARAM_LOGGING,
        )
        exp_handler.run()
        successfull += 1 - exp_handler.error

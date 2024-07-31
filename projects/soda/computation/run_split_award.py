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
    successfull = 0
    for config_game, config_learner in experiment_list:
        exp_handler = Experiment(
            PATH_TO_CONFIGS + "game/" + config_game,
            PATH_TO_CONFIGS + "learner/" + config_learner,
            number_runs=NUMBER_RUNS,
            label_experiment="split_award",
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

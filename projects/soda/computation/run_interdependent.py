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
    successfull = 0
    for config_game, config_learner in experiment_list:
        exp_handler = Experiment(
            PATH_TO_CONFIGS + "game/" + config_game,
            PATH_TO_CONFIGS + "learner/" + config_learner,
            number_runs=NUMBER_RUNS,
            label_experiment="interdependent",
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

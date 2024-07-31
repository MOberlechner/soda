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

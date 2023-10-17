from itertools import product
from time import time

from soda.util.experiment import Experiment

PATH_TO_CONFIGS = "configs/"
PATH_TO_EXPERIMENTS = "experiments/test/"

NUMBER_RUNS = 3
LEARNING = (True,)
SIMULATION = (True,)

LOGGING = True
SAVE_STRAT = True
NUMBER_SAMPLES = int(2**22)
ROUND_DECIMALS = 3

EXPERIMENT_TAG = "main"
games = ["all_pay/all_pay.yaml", "contest_game/tullock_contest_linear.yaml"]

learner = [
    "sofw.yaml",
]

experiment_list = list(product(games, learner))

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
            EXPERIMENT_TAG,
        )
        exp_handler.run()
    t1 = time()
    print(f"\nExperiments finished ({(t1-t0)/60:.1f} min)".ljust(100, "."), "\n")

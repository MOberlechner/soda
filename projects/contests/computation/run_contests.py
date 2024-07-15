from itertools import product
from time import time

from projects.contests.config_exp import *
from soda.util.experiment import Experiment

game_contests = [
    # symmetric, different csf parameters
    "tullock/bidder2_csf1.yaml",
    "tullock/bidder2_csf2.yaml",
    "tullock/bidder2_csf3.yaml",
    "tullock/bidder2_csf4.yaml",
    # asymmetric, different strong agents
    "tullock/bidder2_asym1.yaml",
    "tullock/bidder2_asym2.yaml",
    "tullock/bidder2_asym3.yaml",
    "tullock/bidder2_asym4.yaml",
]
learner_contests = [
    "soda2_eta10_beta05.yaml",
]
experiment_list = list(product(game_contests, learner_contests))

SIMULATION = False
NUMBER_RUNS = 10

if __name__ == "__main__":
    print(f"\nRunning {len(experiment_list)} Experiments".ljust(100, "."), "\n")
    t0 = time()
    successfull = 0
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
            experiment_tag="tullock",
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

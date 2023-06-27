"""
Run this script to run experiments from arxiv paper:

COMPUTING BAYES NASH EQUILIBRIUM STRATEGIES IN AUCTION GAMES VIA SIMULTANEOUS ONLINE DUAL AVERAGING
Martin Bichler, Maximilian Fichtl, Matthias Oberlechner (2023) https://arxiv.org/abs/2208.02036    

Run this script from within the directories "soda/experiments/arxiv"
"""
import itertools
import sys

import pandas as pd

sys.path.append("../../")

from main import main

if __name__ == "__main__":

    # path to config files and experiment directory
    path_config = "experiments/arxiv/configs/"
    path_exp = "experiments/arxiv/"

    # list of all experiments
    experiment_df = pd.read_csv("list_experiments.csv")
    experiment_list = list(experiment_df.itertuples(index=False, name=None))

    number_runs = 1  # each experiment is repeated number_runs times
    learning = True  # compute strategies
    simulation = True  # compute metrics (e.g. L2, util loss, ...) for simulated continuous auctions
    logging = True  # log results from learning and simulation

    number_samples = int(2**22)
    save_strat = True
    save_init_strat = False
    round_decimal = 3

    main(
        path_config,
        path_exp,
        experiment_list,
        number_runs,
        learning,
        simulation,
        logging,
        number_samples,
        save_strat,
        save_init_strat,
        round_decimal,
    )

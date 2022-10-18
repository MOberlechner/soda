import sys

sys.path.append("../../")

import numpy as np

from run_learner import run_experiment

"""
To reproduce the results from the paper, start script from within the directory data_paper/nrl_vs
"""

# List of experiments (setting, experiment_config, learner_config)
experiments = [
    ### single item
    ("single_item", "fpsb32", "poga_random"),
    ("single_item", "fpsb64", "poga_random"),
    # ("single_item", "fpsb32", "soda_random"),
    # ("single_item", "fpsb64", "soda_random"),
    # ("single_item", "fpsb32", "frank_wolfe_random"),
    # ("single_item", "fpsb64", "frank_wolfe_random"),
]

# path to store results
path = ""
# path to config files for experiments
path_config = path + "experiment/stability/configs/"

logging = True  # log results
num_runs = 1000  # repeat each experiment
save_strat = False

# Compute Strategies
for setting, experiment, learn_alg in experiments:
    run_experiment(
        learn_alg, setting, experiment, logging, save_strat, num_runs, path, path_config
    )

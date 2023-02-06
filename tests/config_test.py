""" 
This module tests if all config files are feasible
"""
import os

import numpy as np
import pytest

from src.game import Game
from src.learner.learner import Learner
from src.util.config import Config

# get all config files form soda/configs for games and learners
path_into_project = os.getcwd().split("soda")[0] + "soda/"
path_to_config = "configs/"
testdata_experiments, testdata_learner = [], []
for mechanism_type in os.listdir(f"{path_into_project}{path_to_config}"):
    for experiment in os.listdir(
        f"{path_into_project}{path_to_config}{mechanism_type}/"
    ):
        if experiment != "learner":
            testdata_experiments += [(mechanism_type, experiment.replace(".yaml", ""))]

    for learn_alg in os.listdir(
        f"{path_into_project}{path_to_config}{mechanism_type}/learner/"
    ):
        testdata_learner += [(mechanism_type, learn_alg.replace(".yaml", ""))]


@pytest.mark.parametrize("mechanism_type, experiment", testdata_experiments)
def test_config_games(mechanism_type, experiment):
    config = Config()
    config.get_path("configs/")
    config.get_config_game(mechanism_type, experiment)
    game = config.create_game()

    assert isinstance(game, Game)


@pytest.mark.parametrize("mechanism_type, learn_alg", testdata_learner)
def test_config_learner(mechanism_type, learn_alg):
    config = Config()
    config.get_path("configs/")
    config.get_config_learner(mechanism_type, learn_alg)
    learner = config.create_learner()

    assert isinstance(learner, Learner)

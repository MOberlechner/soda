""" 
This module tests if all config files are feasible
"""
import os
from itertools import product
from typing import Tuple

import numpy as np
import pytest

from src.game import Game
from src.learner.learner import Learner
from src.util.config import Config

""" Add paths to config directories """
path_to_config_dir = ["config/"]


def get_all_configs(dir_config: str) -> tuple:
    """Returns a list of all config files for settings and learners within a specified directory
    We assume the following structure
    -path_to_config_dir:    - game
                                - mechanism1
                                    - config_file1
                                    - config_file2
                                - mechanism2
                                    - config_file3
                                    - config_file4
                                ...
                            - learner
                                - config_file5
                                - config_file6
                                ...

    """
    list_settings, list_learner = [], []
    for root, _, files in os.walk(dir_config):
        if root.split("/")[-2] == "game":
            for f in files:
                list_settings.append(os.path.join(root, f))
        elif root.split("/")[-1] == "learner":
            for f in files:
                list_learner.append(os.path.join(root, f))
    return list_settings, list_learner


for dir_config in path_to_config_dir:
    list_settings, list_learner = get_all_configs(dir_config="configs/")
    test_data = product(list_settings, list_learner)

    @pytest.mark.parametrize("config_game, config_learner", test_data)
    def test_config(config_game, config_learner):
        config = Config(config_game, config_learner)
        game, learner = config.create_setting()
        assert isinstance(game, Game)
        assert isinstance(learner, Learner)

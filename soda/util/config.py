import os
from importlib import import_module
from pathlib import Path
from typing import Dict, Tuple, Union

import yaml
from yaml.loader import SafeLoader

from soda.game import Game
from soda.learner.learner import Learner
from soda.mechanism.mechanism import Mechanism
from soda.strategy import Strategy


class Config:
    """Config Class to create mechanism, approximation game and learner.

    Methods (main):
        create_setting()            creates the setting (game, learner) from config files
        create_strategies()         creates strategies using the initialization method from the config files
    """

    def __init__(
        self, config_game: Union[dict, str], config_learner: Union[dict, str]
    ) -> None:
        """Initialize config class with respective config files

        Args:
            config_game (Union[dict, str]): config file or path to config file for game/mechanism
            config_learner (Union[dict, str]): config file or path to config file for learner

        """
        # config_game
        if isinstance(config_game, dict):
            self.config_game = config_game
        elif isinstance(config_game, str):
            self.config_game = import_config_file(config_game)
        else:
            raise TypeError("config_game should be dict or str (path to config file)")

        # config_learner
        if isinstance(config_learner, dict):
            self.config_learner = config_learner
        elif isinstance(config_learner, str):
            self.config_learner = import_config_file(config_learner)
        else:
            raise TypeError(
                "config_learner should be dict or str (path to config file)"
            )

        # check input
        keys_game = [
            "mechanism",
            "bidder",
            "o_space",
            "a_space",
            "param_prior",
            "param_util",
            "n",
            "m",
        ]
        check_keys_in_config(self.config_game, keys_game, "(game/mechanism)")
        keys_learner = [
            "learner",
            "max_iter",
            "tol",
            "stop_criterion",
            "parameter",
        ]
        check_keys_in_config(self.config_learner, keys_learner, "(learner)")

    def create_setting(self) -> Tuple[Game, Learner]:
        """Create game (from mechanism) and learner from config files"""
        game = self.create_game()
        learner = self.create_learner()
        return game, learner

    def create_strategies(
        self, game: Game, init_method, init_param: Dict = {}
    ) -> Dict[str, Strategy]:
        """create strategies init method from learner config file"""
        strategies = {}
        for i in game.set_bidder:
            strategies[i] = Strategy(i, game)
            strategies[i].initialize(init_method, init_param)
        return strategies

    def create_game(self) -> Game:
        """create game from game config file"""
        n, m = self.config_game["n"], self.config_game["m"]
        mechanism = self.create_mechanism()
        return Game(mechanism, n, m)

    def create_mechanism(self) -> Mechanism:
        """create mechanism from game config file"""
        mechanism = import_attribute(self.config_game["mechanism"])
        keys = ["bidder", "o_space", "a_space", "param_prior", "param_util"]
        args = [self.config_game[key] for key in keys]
        return mechanism(*args)

    def create_learner(self) -> Learner:
        """create learner from learner config file"""
        learner = import_attribute(self.config_learner["learner"])
        keys = ["max_iter", "tol", "stop_criterion", "parameter"]
        args = [self.config_learner[key] for key in keys]
        return learner(*args)


# --------------------------- helper functions ---------------------------


def import_attribute(import_string: str):
    """Imports an attribute from a module by a string reference, e.g. 'numpy.ndarray'"""
    module_name, _, attribute_name = import_string.rpartition(".")
    module = import_module(module_name)
    return getattr(module, attribute_name)


def import_config_file(path: str) -> dict:
    """Import config file from given path"""
    assert path.split(".")[-1] == "yaml", "include .yaml to the config file"
    if os.path.exists(path):
        with open(path) as f:
            return yaml.load(f, Loader=SafeLoader)
    else:
        raise ValueError(f"File {path} does not exist")


def check_keys_in_config(config: dict, keys: list, tag: str = "") -> None:
    """Check if key is config file and raise ValueError if not"""
    for key in keys:
        if key not in config:
            raise ValueError(f"{key} is missing on config {tag}")

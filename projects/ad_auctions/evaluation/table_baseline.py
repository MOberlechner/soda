import os
from itertools import product
from typing import Dict

import numpy as np
import pandas as pd

from projects.ad_auctions.config_exp import PATH_TO_CONFIGS, PATH_TO_EXPERIMENTS
from soda.game import Game
from soda.strategy import Strategy
from soda.util.config import Config


def get_results(
    config_game: dict,
    config_learner: dict,
    path_to_configs: str,
    path_to_experiment: str,
    experiment_tag: str,
    run: int = 0,
):
    """Import computed strategies for a given experiment

    Args:
        config_game (dict): config file for game
        config_learner (dict): config file for learner
        path_to_configs (str): path to config files
        path_to_experiment (str): path to experiments
        experiment_tag (str): experiment tag i.e. subdirectory in experiments
        run (int, optional): Run of experiment. Defaults to 0.
    """

    config = Config(
        path_to_configs + "game/" + config_game,
        path_to_configs + "learner/" + config_learner,
    )
    game, learner = config.create_setting()
    strategies = config.create_strategies(game)

    learner_name = config_learner.split(".")[0]
    game_name = config_game.split(".")[0]
    name = f"{learner_name}_{game_name}_run_{run}"
    path = os.path.join(path_to_experiment, "strategies", experiment_tag)

    for i in strategies:
        strategies[i] = Strategy(i, game)
        strategies[i].load(name, path, load_init=False)

    return game, learner, strategies


def get_revenue(game: Game, strategies: Dict[str, Strategy]) -> float:
    """Compute expected revenue given a strategy profule

    Args:
        game (Game): discretized game
        strategies (Dict[str, Strategy]): dict of computed strategies

    Returns:
        float: expected revenue
    """
    method_revenue = getattr(game.mechanism, "revenue", None)
    if not callable(method_revenue):
        raise NotImplementedError(
            f"revenue() not implemented for mechanism {game.mechanism.name}"
        )

    bid_profiles = game.create_all_bid_profiles()
    revenue = game.mechanism.revenue(bid_profiles).reshape([game.m] * game.n_bidder)

    indices = "".join([chr(ord("a") + i) for i in range(game.n_bidder)])
    for i in range(game.n_bidder):
        idx = chr(ord("a") + i)
        indices += f",{idx}"
    indices += "->"

    return np.einsum(
        indices, revenue, *[strategies[i].x.sum(axis=0) for i in game.bidder]
    )


def create_table(games: list, learner: list) -> pd.DataFrame:
    """create table with expected revenue of all experiments

    Args:
        games (list): list of game configs
        learner (list): list of learner configs

    Returns:
        pd.DataFrame: table
    """
    df = pd.DataFrame(columns=["game", "learner", "revenue"])
    experiment_list = list(product(games, learner))
    for config_game, config_learner in experiment_list:
        game, _, strategies = get_results(
            config_game,
            config_learner,
            PATH_TO_CONFIGS,
            PATH_TO_EXPERIMENTS,
            EXPERIMENT_TAG,
            run=0,
        )
        rev = get_revenue(game, strategies)
        data = [
            {
                "game": config_game.replace(".yaml", ""),
                "learner": config_learner.replace(".yaml", ""),
                "revenue": round(rev, 4),
            }
        ]
        df = pd.concat([df, pd.DataFrame(data)])
    return df


if __name__ == "__main__":

    EXPERIMENT_TAG = "baseline"
    PATH_SAVE = "projects/ad_auctions/results/"
    os.makedirs(PATH_SAVE, exist_ok=True)

    games = [
        "baseline_ql_fp_3.yaml",
        "baseline_ql_sp_3.yaml",
        "baseline_roi_fp_3.yaml",
        "baseline_roi_sp_3.yaml",
        "baseline_rosb_fp_3.yaml",
        "baseline_rosb_sp_3.yaml",
    ]

    learner = [
        "soma2_baseline.yaml",
    ]

    df = create_table(games, learner)
    table = df.to_markdown(
        index=False, tablefmt="pipe", colalign=["center"] * len(df.columns)
    )
    print(f"\nTABLE BASELINE\n\n{table}\n")

    df.to_csv(os.path.join(PATH_SAVE, "table_baseline.csv"), index=False)

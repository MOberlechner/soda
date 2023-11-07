import os
from itertools import product
from typing import Dict

import numpy as np
import pandas as pd

from projects.ad_auctions.config_exp import (
    PATH_SAVE,
    PATH_TO_CONFIGS,
    PATH_TO_EXPERIMENTS,
)
from soda.game import Game
from soda.strategy import Strategy
from soda.util.config import Config
from soda.util.evaluation import get_results


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
                "game": os.path.basename(config_game).replace(".yaml", ""),
                "learner": config_learner.replace(".yaml", ""),
                "revenue": round(rev, 4),
            }
        ]
        df = pd.concat([df, pd.DataFrame(data)])
    return df


if __name__ == "__main__":

    EXPERIMENT_TAG = "baseline"
    os.makedirs(PATH_SAVE, exist_ok=True)

    games = [
        "baseline/ql_fp_3.yaml",
        "baseline/ql_sp_3.yaml",
        "baseline/roi_fp_3.yaml",
        "baseline/roi_sp_3.yaml",
        "baseline/rosb_fp_3.yaml",
        "baseline/rosb_sp_3.yaml",
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

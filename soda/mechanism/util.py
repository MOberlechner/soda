from typing import Dict

import numpy as np
from scipy.special import binom

from soda.game import Game
from soda.strategy import Strategy


def get_allocation_single_item(
    bids: np.ndarray, idx: int, tie_breaking: str, zero_wins: bool = False
) -> np.ndarray:
    """Compute allocation in a single-item auction

    Args:
        bids (np.ndarray): bidding profile
        idx (int): index of agent
        tie_breaking (str): tie-breaking rule (random or lose)
        zero_wins (bool): possible to win with zero bids (only for random tie-breaking). Defaults to False.

    Returns:
        np.ndarray: allocation
    """
    if tie_breaking == "random":
        is_winner = np.where(
            bids[idx] >= np.delete(bids, idx, 0).max(axis=0),
            1,
            0,
        )
        if not zero_wins:
            is_winner = is_winner * np.where(bids[idx] > 0, 1.0, 0.0)
        num_winner = (bids.max(axis=0) == bids).sum(axis=0)
        return is_winner / num_winner
    elif tie_breaking == "lose":
        is_winner = np.where(
            (bids[idx] > np.delete(bids, idx, 0).max(axis=0)) & (bids[idx] > 0),
            1,
            0,
        )
        return is_winner
    else:
        raise ValueError(f"tie_breaking rule {tie_breaking} unknown")


def compute_probability_winning(
    game: Game,
    strategies: Dict[str, Strategy],
    agent: str,
    zero_wins: bool = False,
) -> np.ndarray:
    """Compute probability of winning for each bid given the symmetric (!) strategies

    Args:
        game (Game): approximation game
        strategies (Dict[str, Strategy]): strategy profile
        agent (str): bidder
        zero_wins (bool): possible to win with zero bids (only for random tie-breaking). Defaults to False.

    Raises:
        ValueError: wront tie breaking rule

    Returns:
        np.ndarray: probability of winning
    """

    pdf = strategies[agent].x.sum(axis=0)
    cdf = np.insert(pdf, 0, 0.0).cumsum()[:-1]
    prob_win = cdf ** (game.n_bidder - 1)

    if game.mechanism.tie_breaking == "lose":
        pass
    elif game.mechanism.tie_breaking == "random":
        prob_win += sum(
            [
                binom(game.n_bidder - 1, i)
                * cdf ** (game.n_bidder - i - 1)
                / (i + 1)
                * pdf**i
                for i in range(1, game.n_bidder)
            ]
        )
    else:
        raise ValueError(f"Tie-breaking rule {game.mechanism.tie_breaking} unknown")

    if not zero_wins:
        prob_win[0] = 0.0

    return prob_win

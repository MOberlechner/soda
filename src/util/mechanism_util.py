from typing import Dict, List

import numpy as np
from scipy.special import binom

from src.game import Game
from src.strategy import Strategy


def get_allocation_single_item(
    bids: np.ndarray, idx: int, tie_breaking: str
) -> np.ndarray:
    """Compute allocation in a single-item auction

    Args:
        bids (np.ndarray): bidding profile
        idx (int): index of agent
        tie_breaking (str): tie-breaking rule (random or lose)

    Returns:
        np.ndarray: allocation
    """
    if tie_breaking == "random":
        is_winner = np.where(
            (bids[idx] >= np.delete(bids, idx, 0).max(axis=0)) & (bids[idx] > 0),
            1,
            0,
        )
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


def compute_allocation_lose_1d(bids: np.ndarray, idx: int) -> np.ndarray:
    """Compute allocation for agent with idx i using the lose tie-breaking rule, i.e., in case of ties, nobody wins

    Args:
        bids (np.ndarray): bidding profile
        idx (int): index of agent

    Returns:
        np.ndarray: allocation
    """
    is_winner = np.where(
        (bids[idx] > np.delete(bids, idx, 0).max(axis=0)) & (bids[idx] > 0),
        1,
        0,
    )
    return is_winner


def compute_probability_winning(
    game: Game, strategies: Dict[str, Strategy], agent: str
) -> np.ndarray:
    """Compute probability of winning for each bid given the symmetric (!) strategies

    Args:
        game (Game): approximation game
        strategies (Dict[str, Strategy]): strategy profile
        agent (str): bidder

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

    return prob_win

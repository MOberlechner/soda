from typing import Dict

import numpy as np

from src.game import Game
from src.learner.gradient import Gradient
from src.mechanism.mechanism import Mechanism
from src.strategy import Strategy


def compute_l2_norm(
    mechanism: Mechanism, strategies: Dict[str, Strategy], n_type: int
) -> Dict[str, float]:
    """Computes the approximated L2 norm between computed strategy and analytical BNE

    Args:
        mechanism (Mechanism):
        strategies (Dict[str, Strategy]):
        n_type (int): number of sampled types

    Returns:
        Dict[str, float]
    """
    l2_norm = {}
    for i in mechanism.set_bidder:
        idx = mechanism.bidder.index(i)

        # get observations
        obs = mechanism.sample_types(n_type)

        # get bids & bne for agent i
        bids = strategies[i].bid(obs[idx])
        bne = mechanism.get_bne(i, obs[idx])

        if bne is None:
            l2_norm[i] = -1
            print("BNE is unknown")
        else:
            l2_norm[i] = np.sqrt(1 / n_type * ((bids - bne) ** 2).sum())

    return l2_norm


def compute_utility(
    mechanism: Mechanism, strategies: Dict[str, Strategy], n_type: int
) -> Dict[str, float]:
    """Computes Utility of BNE vs. BNE and Strategy vs. BNE as well as relative utility loss for all agents

    Args:
        mechanism (Mechanism):
        strategies (Dict[str, Strategy]):
        n_type (int): number of sampled types

    Returns:
        Dict[str, float]
    """

    util_bne, util_vs_bne = {}, {}
    for i in mechanism.set_bidder:

        idx = mechanism.bidder.index(i)

        # get bids (for i) and bne (for all)
        obs = mechanism.sample_types(n_type)
        bids = strategies[i].bid(obs[idx])
        bne = np.array(
            [
                mechanism.get_bne(mechanism.bidder[i], obs[i])
                for i in range(mechanism.n_bidder)
            ]
        )

        valuations = mechanism.get_valuation(obs, bids, idx)
        # get utility in bne
        util_bne[i] = mechanism.utility(valuations, bne, idx).mean()
        # get utility vs. bne
        bne[idx] = bids
        util_vs_bne[i] = mechanism.utility(valuations, bne, idx).mean()

    # compute utility loss
    util_loss = {i: 1 - util_vs_bne[i] / util_bne[i] for i in strategies}

    return util_bne, util_vs_bne, util_loss


def compute_util_loss_scaled(
    mechanism: Mechanism, game: Game, strategies: Dict[str, Strategy]
) -> Dict[str, float]:
    """Compute relative utility loss in discretized game

    Args:
        mechanism_scaled (Mechanism): mechanism
        game_scaled (Game): approximation game (discretized mechanism)
        strategies_scaled (Dict[str, Strategy]): strategy profile

    Returns:
        Dict[str, float]: util loss for each agent
    """

    # create learner and prepare gradient computation
    gradient = Gradient()
    if not mechanism.own_gradient:
        gradient.prepare(mechanism, game, strategies)

    util_loss = {}
    for i in mechanism.set_bidder:
        gradient.compute(mechanism, game, strategies, i)
        strategies[i].update_utility_loss()
        util_loss[i] = strategies[i].utility_loss[-1]

    return util_loss

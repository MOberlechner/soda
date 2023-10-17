from typing import Dict

import numpy as np
from scipy.stats import wasserstein_distance
from src.game import Game
from src.learner.dual_averaging import SODA
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


def monotonicity(strategies, game):
    """Check all iterates for monotonicity w.r.t. previous iterate
    < v(s)-v(s'), s-s' > <= 0

    Parameters
    ----------
    strategies : Strategy
    game : Game

    Returns
    -------
    np.ndarray, result for each iteration
    """
    iter = len(strategies[list(strategies.keys())[0]].history)
    return np.array(
        [
            sum(
                [
                    (
                        (
                            strategies[i].history_gradient[t]
                            - strategies[i].history_gradient[t + 1]
                        )
                        * (strategies[i].history[t] - strategies[i].history[t + 1])
                    ).sum()
                    for i in game.bidder
                ]
            )
            for t in range(iter - 1)
        ]
    )


def variational_stability(
    strategies, game, exact_bne: bool = False, normed: bool = False
):
    """Check all iterates for variational stability w.r.t. equilibrium (i.e., last iterate)
    < v(s_t), s_t-s* > <= 0

    Args:
        strategies (_type_): _description_
        game (_type_): Game
        exact_bne (bool, optional): w.r.t last iterate or exact bne (FPSB 2 Bidder uniform). Defaults to False.
        normed (bool, optional): Divide inner product by product of norms. Defaults to False.

    Returns:
        np.ndarray: result for each iteration
    """
    iter = len(strategies[list(strategies.keys())[0]].history)
    if exact_bne:
        bne = {
            i: get_bne_fpsb(strategies[list(strategies.keys())[0]].n)
            for i in strategies
        }
    else:
        bne = {i: strategies[i].x for i in strategies}

    if normed:
        fct_vs = lambda grad, x, bne: (grad * (x - bne)).sum() / max(
            np.linalg.norm(grad) * np.linalg.norm(x - bne), 1e-20
        )
    else:
        fct_vs = lambda grad, x, bne: (grad * (x - bne)).sum()

    return np.array(
        [
            sum(
                [
                    fct_vs(
                        strategies[i].history_gradient[t],
                        strategies[i].history[t],
                        bne[i],
                    )
                    for i in game.bidder
                ]
            )
            for t in range(iter - 1)
        ]
    )


def best_response_stability(
    strategies, game, exact_bne: bool = False, normed: bool = False
):
    """Check if br points towards equilibrium (i.e., last iterate)
    < br(s_t)-s_t, s_t-s* > <= 0

    Args:
        strategies (_type_): _description_
        game (_type_): Game
        exact_bne (bool, optional): w.r.t last iterate or exact bne (FPSB 2 Bidder uniform). Defaults to False.
        normed (bool, optional): Divide inner product by product of norms. Defaults to False.

    Returns:
        np.ndarray: result for each iteration
    """
    iter = len(strategies[list(strategies.keys())[0]].history)
    if exact_bne:
        bne = {
            i: get_bne_fpsb(strategies[list(strategies.keys())[0]].n)
            for i in strategies
        }
    else:
        bne = {i: strategies[i].x for i in strategies}

    if normed:
        fct_brs = lambda br, x, bne: ((br - x) * (x - bne)).sum() / max(
            np.linalg.norm(br - x) * np.linalg.norm(x - bne), 1e-20
        )
    else:
        fct_brs = lambda br, x, bne: ((br - x) * (x - bne)).sum()

    return np.array(
        [
            sum(
                [
                    fct_brs(
                        strategies[i].history_best_response[t],
                        strategies[i].history[t],
                        bne[i],
                    )
                    for i in game.bidder
                ]
            )
            for t in range(iter - 1)
        ]
    )


def next_iterate_stability(
    strategies, game, exact_bne: bool = False, normed: bool = False
):
    """Check if update points towards equilibrium (i.e., last iterate)
    < s_t+1-s_t, s_t-s* > <= 0

    Args:
        strategies (_type_):
        game (_type_): Game
        exact_bne (bool, optional): w.r.t last iterate or exact bne (FPSB 2 Bidder uniform). Defaults to False.
        normed (bool, optional): Divide inner product by product of norms. Defaults to False.

    Returns:
        np.ndarray: result for each iteration
    """
    iter = len(strategies[list(strategies.keys())[0]].history)
    if exact_bne:
        bne = {
            i: get_bne_fpsb(strategies[list(strategies.keys())[0]].n)
            for i in strategies
        }
    else:
        bne = {i: strategies[i].x for i in strategies}

    if normed:
        fct_nis = lambda x_next, x_now, bne: (
            (x_next - x_now) * (x_now - bne)
        ).sum() / max(
            np.linalg.norm(x_next - x_now) * np.linalg.norm(x_now - bne), 1e-20
        )
    else:
        fct_nis = lambda x_next, x_now, bne: ((x_next - x_now) * (x_now - bne)).sum()

    return np.array(
        [
            sum(
                [
                    fct_nis(
                        strategies[i].history[t + 1], strategies[i].history[t], bne[i]
                    )
                    for i in game.bidder
                ]
            )
            for t in range(iter - 1)
        ]
    )


def gradient_direction(strategies, game, normed: bool = False):
    """Check if gradient of each update points towards gradient of equilibrium (i.e., last iterate)
    < v(s), v(s^*) > <= 0

    Args:
        strategies (_type_):
        game (_type_): Game
        normed (bool, optional): Divide inner product by product of norms. Defaults to False.

    Returns:
        np.ndarray: result for each iteration
    """
    iter = len(strategies[list(strategies.keys())[0]].history)

    if normed:
        fct_gs = lambda x, y: (x * y).sum() / max(
            np.linalg.norm(x) * np.linalg.norm(y), 1e-20
        )
    else:
        fct_gs = lambda x, y: (x * y).sum()

    return np.array(
        [
            sum(
                [
                    fct_gs(
                        np.vstack(
                            [strategies[i].history_gradient[t] for i in game.bidder]
                        ),
                        np.vstack(
                            [strategies[i].history_gradient[-1] for i in game.bidder]
                        ),
                    )
                ]
            )
            for t in range(iter - 1)
        ]
    )


def gradient_distance(strategies, game):
    """Check if gradient of each update points towards gradient of equilibrium (i.e., last iterate)
     norm(v(s_t)-v(s^*))

    Args:
        strategies (_type_):
        game (_type_): Game

    Returns:
        np.ndarray: result for each iteration
    """
    iter = len(strategies[list(strategies.keys())[0]].history)
    fct_gd = lambda x, y: np.linalg.norm(x - y) ** 2

    return np.array(
        [
            np.sqrt(
                sum(
                    [
                        fct_gd(
                            strategies[i].history_gradient[t],
                            strategies[i].history_gradient[-1],
                        )
                        for i in game.bidder
                    ]
                )
            )
            for t in range(iter - 1)
        ]
    )


def wasserstein_dist(strategies, game):
    """Compute Wasserstein Distance to last iterate
    Wasserstein distance is computed for each mixed strategy (fixed observation) separately and then averaged over
    and summed over all bidders

    Args:
        strategies (_type_): Strategy
        game (_type_): Game

    Returns:
        np.ndarray: result for each iteration
    """
    iter = len(strategies[list(strategies.keys())[0]].history)
    fct_wd = lambda x, y, a_discr: np.mean(
        [wasserstein_distance(x[j], y[j], a_discr, a_discr) for j in range(game.n)]
    )

    return np.array(
        [
            np.sqrt(
                sum(
                    [
                        fct_wd(
                            strategies[i].history_gradient[t],
                            strategies[i].history_gradient[-1],
                            strategies[i].a_discr,
                        )
                        for i in game.bidder
                    ]
                )
            )
            for t in range(iter - 1)
        ]
    )


def get_bne_fpsb(n: int, odd: bool = True) -> np.ndarray:
    """Discrete BNE for FPSB unform prior, 2 bidder

    Args:
        n (int): number of discretization points
        odd (bool, optional): which BNE. Defaults to True.

    Returns:
        np.ndarray: bne
    """

    # bne_odd
    if odd:
        bne = np.vstack(
            [
                [0] * (n // 2 - 1),
                (np.kron(np.eye(n // 2 - 1), np.ones(2)).T),
                [0] * (n // 2 - 1),
            ]
        )
        bne = np.hstack(
            [np.eye(1, n, 0).reshape(n, 1), bne, np.eye(1, n, n - 1).reshape(n, 1)]
        )
        x = np.hstack([bne, np.zeros((n, n // 2 - 1))])

    else:
        # bne even
        x = np.hstack([np.kron(np.eye(n // 2), np.ones(2)).T, np.zeros((n, n // 2))])
    return x / x.sum()

from time import sleep, time
from typing import Dict, List

import numpy as np
from opt_einsum import contract, contract_path
from tqdm import tqdm


def dual_averaging(
    game,
    strategies: Dict,
    max_iter: int,
    steprule_bool: bool,
    tol: float,
    eta: float,
    beta: float,
):
    """
    Simultanous online dual averaging

    Parameters
    ----------
    game : Class Game,
    strategies :
    max_iter :
    steprule_bool
    tol :
    eta :
    beta :

    Returns
    -------
    """

    # dont ask why, but it improves performance significantly
    u_dict = game.utility.copy()

    # prepare gradients
    indices, path = prepare_grad(game, strategies)
    gradients = {}

    # init variables
    convergence = False
    min_max_util_loss = 1
    t_max = 0

    for t in tqdm(
        range(int(max_iter)),
        unit_scale=True,
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}",
    ):

        # compute gradients
        for i in game.set_bidder:
            gradients[i] = compute_gradient(
                strategies,
                game.bidder,
                i,
                game.utility[i],
                game.weights,
                indices[i],
                path[i],
            )

        # update utility, utility loss
        for i in game.set_bidder:
            strategies[i].update_utility(gradients[i])
            strategies[i].update_utility_loss(gradients[i])

        # check convergence
        max_util_loss = np.max(
            [strategies[i].utility_loss[-1] for i in game.set_bidder]
        )
        min_max_util_loss = (
            max_util_loss if max_util_loss < min_max_util_loss else min_max_util_loss
        )
        if max_util_loss < tol:
            t_max = t
            convergence = True
            break

        # update strategy
        for i in game.set_bidder:
            stepsize = step_rule(
                steprule_bool, eta, beta, t, gradients[i], strategies[i].dim_o
            )
            strategies[i].update_strategy(gradients[i], stepsize)

    # Print result
    if convergence:
        print("Convergence after", t_max, "iterations")
        print("Relative utility loss", round(max_util_loss * 100, 3), "%")
    else:
        print("No convergence")
        print("Current relative utility loss", round(max_util_loss * 100, 3), "%")
        print("Best relative utility loss", round(min_max_util_loss * 100, 3), "%")


def compute_gradient(
    strategies, bidder, player, u_array, weights, index, path="optimal"
):
    """
    Compute gradient for player i given a strategy profile and utility array.
    This operation is based on an optimized version of np.einsum, namely contract
    (but not sure if difference is significant)

    Parameters
    ----------
    strategies : dict, contains strategies
    bidder : list, of bidders
    player : int/str, player i
    u_array : array, utility array for player i
    weights: array, contains interdependencies of valuations
    index : str, index for contract (from get_indices())
    path : str, optimal or precomputed path for contract()

    Returns
    -------
    array, gradient of player i
    """

    opp = bidder.copy()
    opp.remove(player)

    # bidders observations/valuations are independent
    if weights is None:
        return contract(
            index,
            *[u_array]
            + [
                strategies[i].x.sum(axis=tuple(range(strategies[i].dim_o))) for i in opp
            ],
            optimize=path
        )
    # bidders observations/valuations are correlated
    else:
        return contract(
            index,
            *[u_array] + [strategies[i].x for i in opp] + [weights],
            optimize=path
        )


def step_rule(steprule_bool, eta, beta, t, grad, dim_o):
    """
    Step sizes are not summable. but square-summable

    Parameters
    ----------
    steprule_bool : bool, use decreasing step rule or heuristic
    eta : float, initial stepsize
    beta : float, <= 1
    t : int, iteration
    grad : array, gradient - only necessary for step size heuristic
    dim_o : int, dimension of observation space - only necessary for step size heuristic

    Returns
    -------
    eta : np.ndarray
    """

    if steprule_bool:
        # decreasing step rule (Mertikopoulos&Zhou)
        return np.array(eta * 1 / (t + 1) ** beta)

    else:
        # heuristic
        scale = grad.max(axis=tuple(range(dim_o, len(grad.shape))))
        scale[scale < 0] = 0.001
        scale[scale < 1e-100] = 1e-100
        return eta / scale


def prepare_grad(game, strategies):
    """

    Parameters
    ----------
    game  : class
    strategies : class

    Returns
    -------
    indices : str, contains indices as strings for einsum/contract for each bidder
    path : dict, contains path for
    """
    ind = game.weights is None
    indices, path = {}, {}

    dim_o, dim_a = strategies[game.bidder[0]].dim_o, strategies[game.bidder[0]].dim_a
    n_bidder = game.n_bidder

    # indices of all actions
    act = "".join([chr(ord("a") + i) for i in range(n_bidder * dim_a)])
    # indices of all valuations
    val = "".join([chr(ord("A") + i) for i in range(n_bidder * dim_o)])

    for i in game.set_bidder:

        idx = game.bidder.index(i)
        idx_opp = [i for i in range(n_bidder) if i != idx]

        # indices of utility array
        start = act + val[idx * dim_o : (idx + 1) * dim_o]
        # indices of bidder i's strategy
        end = (
            "->"
            + val[idx * dim_o : (idx + 1) * dim_o]
            + act[idx * dim_a : (idx + 1) * dim_a]
        )

        if ind:
            # valuations are independent
            indices[i] = (
                start
                + ","
                + ",".join([act[j * dim_a : (j + 1) * dim_a] for j in idx_opp])
                + end
            )
            path[i] = contract_path(
                indices[i],
                *[game.utility[i]]
                + [
                    strategies[game.bidder[j]].x.sum(axis=tuple(range(dim_o)))
                    for j in idx_opp
                ],
                optimize="optimal"
            )[0]
        else:
            # valuations are correlated
            indices[i] = (
                start
                + ","
                + ",".join(
                    [
                        val[j * dim_o : (j + 1) * dim_o]
                        + act[j * dim_a : (j + 1) * dim_a]
                        for j in idx_opp
                    ]
                )
                + ","
                + val
                + end
            )
            path[i] = contract_path(
                indices[i],
                *[game.utility[i]]
                + [strategies[game.bidder[j]].x for j in idx_opp]
                + [np.ones([game.n] * (dim_o * n_bidder))],
                optimize="optimal"
            )[0]

    return indices, path

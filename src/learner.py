import numpy as np
from opt_einsum import contract, contract_path
from tqdm import tqdm
from time import sleep
from typing import List, Dict


def dual_averaging(bidder: List, strategies: Dict, payoff_dict: Dict, max_iter: int, tol: float ):
    """
    Simultanous online dual averaging

    Parameters
    ----------
    bidder : list
    strategies :
    u_dict :
    max_iter :
    tol :

    Returns
    -------

    """

    # prepare gradient
    gradient = {}

    # different bidders
    set_bidder = list(set(bidder))

    # init variables
    convergence = False
    min_max_util_loss = 1
    t_max = 0

    for t in tqdm(range(int(max_iter)), unit_scale=True, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'):

        # compute gradients
        for i in set_bidder:
            gradient[i] = gradient(strategies, bidder,  i, payoff_dict, weights, indices, path='optimal')

        # update utility, utility loss
        for i in set_bidder:
            strategies[i].update_utility(gradient[i])
            strategies[i].update_utility_loss(gradient[i])

        # check convergence
        max_util_loss = np.max([strategies[i].utility_loss[-1] for i in set_bidder])
        min_max_util_loss = max_util_loss if max_util_loss < min_max_util_loss else min_max_util_loss
        if max_util_loss < tol:
            t_max = t
            convergence = True
            break

    # Compute L2 norm to last iterate
    distance_to_strategy = {i: [np.linalg.norm(strategies[i].x - y) for y in strategies[i].history] for i in set_bidder}

    # Print result
    sleep(0.05)
    if convergence:
        print('Convergence after', t_max, 'iterations')
        print('Relative utility loss', round(max_util_loss*100, 3), '%')
    else:
        print('No convergence')
        print('Current relative utility loss', round(max_util_loss*100, 3), '%')
        print('Best relative utility loss', round(min_max_util_loss*100, 3), '%')

    return strategies, distance_to_strategy


def gradient(strategies, bidder,  player, u_array, weights, indices, path='optimal'):
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
    indices : str, indices for contract (from get_indices())
    path : str, optimal or precomputed path for contract()

    Returns
    -------
    array, gradient of player i
    """

    opp = bidder.copy()
    opp.remove(player)

    # bidders observations/valuations are independent
    if weights is None:
        return contract(indices, *[u_array] + [strategies[i].sum(axis=tuple(range(strategies[i].dim_o)))
                                               for i in opp], optimize=path)
    # bidders observations/valuations are correlated
    else:
        return contract(indices, *[u_array] + [strategies[i].x for i in opp] + [weights], optimize=path)







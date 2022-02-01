import numpy as np
from opt_einsum import contract, contract_path
from tqdm import tqdm
from time import sleep
from typing import List, Dict


def dual_averaging(game, strategies: Dict, max_iter: int, steprule_bool: bool, tol: float, eta: float, beta: float):
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

    # prepare gradients
    indices, path = prepare_grad(game, strategies):
    gradients = {}

    # different bidders
    set_bidder = list(set(bidder))

    # init variables
    convergence = False
    min_max_util_loss = 1
    t_max = 0

    for t in tqdm(range(int(max_iter)), unit_scale=True, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'):

        # compute gradients
        for i in game.set_bidder:
            gradients[i] = compute_gradient(strategies, game.bidder,  i, util_dict, weights, indices, path='optimal')

        # update utility, utility loss
        for i in set_bidder:
            strategies[i].update_utility(gradients[i])
            strategies[i].update_utility_loss(gradients[i])

        # check convergence
        max_util_loss = np.max([strategies[i].utility_loss[-1] for i in set_bidder])
        min_max_util_loss = max_util_loss if max_util_loss < min_max_util_loss else min_max_util_loss
        if max_util_loss < tol:
            t_max = t
            convergence = True
            break

        # update strategy
        for i in set_bidder:
            stepsize = step_rule(steprule_bool, eta, beta,  t, gradients[i], strategies[i].dim_o)
            strategies[i].update_strategy(gradients[i], stepsize)

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


def compute_gradient(strategies, bidder,  player, u_array, weights, indices, path='optimal'):
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


def step_rule(steprule_bool, eta, beta,  t, grad, dim_o):
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
        return np.array(eta * 1/(t+1)**beta)

    else:
        # heuristic
        scale = grad.max(axis=tuple(range(dim_o, len(grad.shape))))
        scale[scale < 0] = 0.001
        scale[scale < 1e-100] = 1e-100
        return eta / scale


def prepare_grad(auction: str, bidder: List, strategies: Dict, u_dict: Dict, fact, dim_o=1, dim_a=1):
    """

    Parameters
    ----------
    auction : str, specify auction
    bidder : list, list of bidders
    strategies : dict, contains all initial strategies
    u_dict : dict, contains utility arrays
    fact : array, weights of observation combinations (or None if independent)
    dim_o : dimension of observation space
    dim_a : dimension of action space

    Returns
    -------
    indices : str, contains indices as strings for einsum/contract for each bidder
    path : dict, contains path for
    """
    ind = fact is None
    indices, path = {}, {}

    n_bidder = len(bidder)
    n = u_dict[bidder[0]].shape[-1]

    # no preparation necessary for single-item FPSB auction with symmetric iid bidders
    if len(u_dict[bidder[0]].shape) == 2:
        return {bidder[0]: ''}, {bidder[0]: [None]}

    # indices of all actions
    act = ''.join([chr(ord('a') + i) for i in range(n_bidder * dim_a)])
    # indices of all valuations
    val = ''.join([chr(ord('A') + i) for i in range(n_bidder * dim_o)])

    for i in set(bidder):

        idx = bidder.index(i)
        idx_opp = [i for i in range(n_bidder) if i != idx]

        # indices of utility array
        start = act + val[idx*dim_o:(idx+1)*dim_o]
        # indices of bidder i's strategy
        end = '->' + val[idx * dim_o:(idx + 1) * dim_o] + act[idx * dim_a:(idx + 1) * dim_a]

        if auction == 'cv':
            # in the correlated value model we distinguish between observation and valuation
            indices[i] = 'opqk,mp,nq,klmn->lo'
            path[i] = contract_path(indices[i], *[u_dict[i]] + [strategies[bidder[j]]
                                            for j in idx_opp] + [np.ones([n] * (n_bidder+1))], optimize='optimal')[0]

        elif ind:
            # valuations are independent
            indices[i] = start + ',' + ','.join([act[j * dim_a:(j + 1) * dim_a] for j in idx_opp]) + end
            path[i] = contract_path(indices[i], *[u_dict[i]] + [strategies[bidder[j]].sum(axis=tuple(range(dim_o)))
                                            for j in idx_opp], optimize='optimal')[0]
        else:
            # valuations are correlated
            indices[i] = start + ',' + ','.join([val[j * dim_o:(j + 1) * dim_o] + act[j * dim_a:(j + 1) * dim_a]
                                            for j in idx_opp]) + ',' + val + end
            path[i] = contract_path(indices[i], *[u_dict[i]] + [strategies[bidder[j]]
                                            for j in idx_opp] + [np.ones([n]*(dim_o*n_bidder))], optimize='optimal')[0]

    return indices, path



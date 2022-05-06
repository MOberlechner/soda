import numpy as np

from src.learner import SODA


def compute_l2_norm(mechanism, strategies, n_obs):
    """Compute L2 distance between computed strategies and BNE

        Parameters
        ----------
        mechanism : class
        strategies : class
        n_obs : int, number of observations sampled

        Returns
        -------
        dict, l2 norm for each strategy
    ö
    """

    l2_norm = {}
    for i in mechanism.set_bidder:
        idx = mechanism.bidder.index(i)

        # get observations
        obs = mechanism.draw_values(n_obs)

        # get bids & bne for agent i
        bids = strategies[i].bid(obs[idx])
        bne = mechanism.get_bne(i, obs[idx])

        if bne is None:
            l2_norm[i] = -1
            print("BNE is unknown")
        else:
            l2_norm[i] = np.sqrt(1 / n_obs * ((bids - bne) ** 2).sum())

    return l2_norm


def compute_utility(mechanism, strategies, n_obs):
    """Compute utilities and utility loss when playing computed strategies against BNE

    Parameters
    ----------
    mechanism : class
    strategies : class
    n_obs : int, number of observations sampled

    Returns
    -------
    dict, utility in BNE, utility and utility loss playing against BNE
    """

    util_bne, util_vs_bne = {}, {}
    for i in mechanism.set_bidder:

        idx = mechanism.bidder.index(i)

        # get bids (for i) and bne (for all)
        obs = mechanism.draw_values(n_obs)
        bids = strategies[i].bid(obs[idx])
        bne = np.array(
            [
                mechanism.get_bne(mechanism.bidder[i], obs[i])
                for i in range(mechanism.n_bidder)
            ]
        )

        if mechanism.values == "private":
            valuations = obs[idx]
        elif mechanism.values == "affiliated":
            valuations = obs
        elif mechanism.values == "common":
            valuations = obs[mechanism.n_bidder]
        else:
            raise NotImplementedError

        # get utility in bne
        util_bne[i] = mechanism.utility(valuations, bne, idx).mean()

        # get utility vs. bne
        bne[idx] = bids
        util_vs_bne[i] = mechanism.utility(valuations, bne, idx).mean()

    # compute utility loss
    util_loss = {i: 1 - util_vs_bne[i] / util_bne[i] for i in strategies}

    return util_bne, util_vs_bne, util_loss


def compute_util_loss_scaled(mechanism_scaled, game_scaled, strategies_scaled):
    """Compute Utility Loss

    Parameters
    ----------
    mechanism_scaled : class, mechanism
    game_scaled : class, discretized game
    strategies_scaled : class, strategy induced by lower dimensional strategy

    Returns
    -------
    Dict, utility loss in larger (scaled) game
    """

    # create learner and prepare gradient computation
    soda = SODA(max_iter=0, tol=0, steprule_bool=True, eta=1, beta=1)
    if not mechanism_scaled.own_gradient:
        soda.prepare_grad(game_scaled, strategies_scaled)

    util_loss_scaled = {}
    for i in mechanism_scaled.set_bidder:
        grad = soda.compute_gradient(strategies_scaled, game_scaled, i)
        strategies_scaled[i].update_utility_loss(grad)
        util_loss_scaled[i] = strategies_scaled[i].utility_loss[-1]

    return util_loss_scaled

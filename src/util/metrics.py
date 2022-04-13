import numpy as np


def compute_l2_norm(mechanism, strategies, n_obs):
    """ Compute L2 distance between computed strategies and BNE

    Parameters
    ----------
    mechanism : class
    strategies : class
    n_obs : int, number of observations sampled

    Returns
    -------
    dict, l2 norm for each strategy

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
    """ Compute utilities and utility loss when playing computed strategies against BNE

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
                mechanism.get_bne(i, obs[mechanism.bidder.index(i)])
                for i in mechanism.bidder
            ]
        )

        # get utility in bne
        util_bne[i] = mechanism.utility(obs[idx], bne, idx).mean()

        # get utility vs. bne
        bne[idx] = bids
        util_vs_bne[i] = mechanism.utility(obs[idx], bne, idx).mean()

        # compute utility loss
        util_loss = {i: 1 - util_vs_bne[i] / util_bne[i] for i in strategies}

        return util_bne, util_vs_bne, util_loss

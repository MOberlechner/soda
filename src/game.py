from itertools import product
from typing import Dict, List

import numpy as np

# -------------------------------------------------------------------------------------------------------------------- #
#                                      CLASS GAME : DISCRETIZED AUCTION GAME                                           #
# -------------------------------------------------------------------------------------------------------------------- #


class Game:
    def __init__(self, mechanism, n: int, m: int):

        self.bidder = mechanism.bidder
        self.set_bidder = mechanism.set_bidder
        self.n_bidder = mechanism.n_bidder
        self.n = n
        self.m = m

        # discrete action and observation space
        self.o_discr = {
            i: discr_spaces(mechanism.o_space[i], n) for i in self.set_bidder
        }
        self.a_discr = {
            i: discr_spaces(mechanism.a_space[i], m, midpoint=False)
            for i in self.set_bidder
        }

        # marginal prior for bidder
        self.prior = {i: self.get_prior(mechanism, i) for i in self.set_bidder}
        self.weights = None

        self.utility = {}

    def get_utility(self, mechanism):

        for i in self.set_bidder:
            idx = self.bidder.index(i)
            # create all possible bids
            bids = np.array(
                [
                    np.array(
                        [
                            self.a_discr[self.bidder[k]][j[k]]
                            for j in product(range(self.m), repeat=self.n_bidder)
                        ]
                    )
                    for k in range(self.n_bidder)
                ]
            )
            # all valuations
            valuations = self.o_discr[i]

            # compute utility
            self.utility[i] = (
                mechanism.utility(valuations, bids, idx)
                .transpose()
                .reshape(tuple([self.m] * self.n_bidder + [self.n]))
            )

    def get_prior(self, mechanism, agent):
        p = mechanism.prior_pdf(self.o_discr[agent], agent)
        return p / p.sum()


# -------------------------------------------------------------------------------------------------------------------- #
#                                                 HELPERFUNCTIONS                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


def discr_spaces(interval: List, n: int, midpoint: bool):
    """
    Parameters
    ----------
    interval : dict, contains lists with observation intervals for each bidder
    n : number of discretization points
    midpoint : take midpoints of discretization

    Returns
    -------
    discrete_space: dict, containing array with discretized space for each bidder
    dim : int, dimension of respective space
    """

    # check dimension of observation space (if more dimensional, interval is nested list)
    if len(np.array(interval).shape) > 1:
        return np.array(
            [discr_interval(interv[0], interv[1], n, midpoint) for interv in interval]
        )
    else:
        return discr_interval(interval[0], interval[1], n, midpoint)


def discr_interval(a: float, b: float, n: int, midpoint: bool):
    """
    Discretize interval [a,b] using n points

    Parameters
    ----------
    a : float, lower bound interval
    b : float, upper bound interval
    n : int, number of discretization points
    midpoint : bool, if True returns midpoints of n subintervals, else a, b and n-2 points in between

    Returns
    -------
    array(n), discretized interval
    """
    if midpoint:
        # split interval in n equally sized subintervals and take midpoints (observation/valuation)
        return a + (0.5 + np.arange(n)) * (b - a) / n
    else:
        # take minimal and maximal value of interval and distribute n-2 remaining points equally in between (action)
        return a + (np.arange(n)) * (b - a) / (n - 1)

from itertools import product
from typing import Dict, List

import numpy as np

from src.mechanism import double_auction

# -------------------------------------------------------------------------------------------------------------------- #
#                                      CLASS GAME : DISCRETIZED AUCTION GAME                                           #
# -------------------------------------------------------------------------------------------------------------------- #


class Game:
    def __init__(
        self,
        mechanism: str,
        bidder: List,
        o_intervals: Dict,
        a_intervals: Dict,
        n: int,
        m: int,
    ):

        self.mechanism = mechanism
        self.bidder = bidder
        self.set_bidder = list(set(bidder))
        self.n_bidder = len(bidder)
        self.n = n
        self.m = m

        # discrete action and observation space
        self.a_discr = {i: discr_spaces(a_intervals[i], m) for i in self.set_bidder}
        self.o_discr = {i: discr_spaces(o_intervals[i], n) for i in self.set_bidder}
        # TODO: what about different observation and valuation spaces

        # marginal prior for bidder
        self.prior = {}
        self.weights = None
        # TODO: include correlations

        self.utility = {}

    def get_utility(self, param):

        utility = {}
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
            if self.mechanism == "double_auction":
                self.utility[i] = (
                    double_auction.util(valuations, bids, self.bidder, idx, param)
                    .transpose()
                    .reshape(tuple([self.m] * self.n_bidder + [self.n]))
                )

    def get_prior(self, param):
        for i in self.set_bidder:
            self.prior[i] = discr_prior(self.o_discr[i], param)


# -------------------------------------------------------------------------------------------------------------------- #
#                                                 HELPERFUNCTIONS                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


def discr_prior(o_discr, param):
    """

    Parameters
    ----------
    o_discr : np.ndarray, discretized observation/valuation space
    param : dict, contains parameter such as distribution, ...

    Returns
    -------
    np.ndarray : prior distribution, shape similar to o_discr
    """
    # TODO: Implement more complex priors

    # check if input is valid
    if "distribution" not in param.keys():
        raise ValueError("Distribution for prior not defined")

    dist = param["distribution"]

    if dist == "uniform":
        p = np.ones(o_discr.shape)

    elif dist == "gaussian":
        if ("mu" in param.keys()) & ("sigma" in param.keys()):
            mu = param["mu"]
            sigma = param["sigma"]
            p = np.exp(-1 / 2 * ((o_discr - mu) / sigma) ** 2)
        else:
            raise ValueError("(mu, sigma) for Gaussian distribution not defined")

    elif dist == "exponential":
        if "lambda" in param.keys():
            lamb = param["lambda"]
            p = lamb * np.exp(-lamb * o_discr)
        else:
            raise ValueError("lambda for exponential distribution not defined")
    else:
        raise ValueError("Unknown distribution for prior", dist)

    return p / p.sum()


def discr_spaces(interval: List, n: int):
    """
    Parameters
    ----------
    interval : dict, contains lists with observation intervals for each bidder
    n : number of discretization points

    Returns
    -------
    discrete_space: dict, containing array with discretized space for each bidder
    dim : int, dimension of respective space
    """

    # check dimension of observation space (if more dimensional, interval is nested list)
    if len(np.array(interval).shape) > 1:
        return np.array(
            [discr_interval(interv[0], interv[1], n) for interv in interval]
        )
    else:
        return discr_interval(interval[0], interval[1], n)


def discr_interval(a: float, b: float, n: int, midpoint: bool = True):
    """
    Discretize interval [a,b] using n points

    Parameters
    ----------
    a : float, lower bound interval
    b : float, upper bound interval
    n : int, number of discretization points
    midpoint : bool, if True returns midpoints of n subintervals, else bounds of intervals (n+1 points)

    Returns
    -------
    array(n), discretized interval
    """
    if midpoint:
        return a + (0.5 + np.arange(n)) * (b - a) / n
    else:
        return a + (np.arange(n + 1)) * (b - a) / n

from itertools import product
from typing import List

import numpy as np

from src.prior import compute_weights, marginal_prior_pdf

# -------------------------------------------------------------------------------------------------------------------- #
#                                      CLASS GAME : DISCRETIZED AUCTION GAME                                           #
# -------------------------------------------------------------------------------------------------------------------- #


class Game:
    def __init__(self, mechanism, n: int, m: int):
        """Given a mechanism and number of discretization points, we can create approximation game by
        discretizating the respective spaces.

        Parameters
        ----------
        mechanism : class, mechanism defines auction game
        n : int, number of discretization points in type space
        m : int, number of discretization points in action space
        """

        self.name = mechanism.name
        self.bidder = mechanism.bidder
        self.set_bidder = mechanism.set_bidder
        self.n_bidder = mechanism.n_bidder
        self.n = n
        self.m = m

        # we distinguish between private, affiliated, common values ... model
        self.values = mechanism.values

        # discrete action and observation space (and optional valuation space)
        self.o_discr = {
            i: discr_spaces(mechanism.o_space[i], n, midpoint=True)
            for i in self.set_bidder
        }
        self.a_discr = {
            i: discr_spaces(mechanism.a_space[i], m, midpoint=False)
            for i in self.set_bidder
        }
        if hasattr(mechanism, "v_space"):
            self.v_discr = {
                i: discr_spaces(mechanism.v_space[i], m, midpoint=True)
                for i in self.set_bidder
            }

        # dimension of spaces
        self.dim_o = (
            1
            if len(self.o_discr[self.bidder[0]].shape) == 1
            else self.o_discr[self.bidder[0]].shape[0]
        )
        self.dim_a = (
            1
            if len(self.a_discr[self.bidder[0]].shape) == 1
            else self.a_discr[self.bidder[0]].shape[0]
        )

        # marginal prior for bidder
        self.prior = {i: self.get_prior(mechanism, i) for i in self.set_bidder}

        self.weights = self.get_weights(mechanism)
        self.utility = {}

    def get_utility(self, mechanism):

        for i in self.set_bidder:

            idx = self.bidder.index(i)
            if self.dim_a == 1:
                # create all possible bids: 1-dim action space
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
            else:
                # create all possible bids: multi-dim action space
                bids = np.array(
                    [
                        np.array(
                            [
                                np.array(
                                    [
                                        self.a_discr[self.bidder[i]][k][
                                            j[self.dim_a * i + k]
                                        ]
                                        for j in product(
                                            range(self.m),
                                            repeat=self.n_bidder * self.dim_a,
                                        )
                                    ]
                                )
                                for k in range(self.dim_a)
                            ]
                        )
                        for i in range(self.n_bidder)
                    ]
                )

            if self.values == "private":
                # valuation only depends on own observation
                valuations = self.o_discr[i]
                self.utility[i] = (
                    mechanism.utility(valuations, bids, idx)
                    .transpose()
                    .reshape(
                        tuple(
                            [self.m] * (self.dim_a * self.n_bidder)
                            + [self.n] * self.dim_o
                        )
                    )
                )

            elif self.values == "affiliated":
                # affiliated values model with correlated observations and common value
                valuations = self.o_discr[i]
                self.utility[i] = (
                    mechanism.utility(valuations, bids, idx)
                    .transpose()
                    .reshape(tuple([self.m] * self.n_bidder + [self.n] * self.n_bidder))
                )

            elif self.values == "common":
                valuations = self.v_discr[i]
                self.utility[i] = (
                    mechanism.utility(valuations, bids, idx)
                    .transpose()
                    .reshape(
                        tuple(
                            [self.m] * (self.dim_a * self.n_bidder)
                            + [self.n] * self.dim_o
                        )
                    )
                )
            else:
                raise ValueError

    def get_prior(self, mechanism, agent: str) -> np.ndarray:
        """Given prior distribution specified in mechanism, returns discretized prior
        If discretized observation space has only one entry, it corresponds to the complete information setting and the probability is equal to 1

        Args:
            mechanism: auction mechanism
            agent (str): agent

        Returns:
            np.ndarray: discretized prior for agent
        """
        if self.o_discr[agent].size == 1:
            p = np.array([1])
        else:
            p = marginal_prior_pdf(mechanism, self.o_discr[agent], agent)
        return p / p.sum()

    def get_weights(self, mechanism):
        return compute_weights(self, mechanism)


# -------------------------------------------------------------------------------------------------------------------- #
#                                                 HELPERFUNCTIONS                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


def discr_spaces(interval: List, n_discrete: int, midpoint: bool) -> np.ndarray:
    """Discretize Spaces (possibly multidimensional)

    Args:
        interval (List): contains lists with space intervals for each bidder
        n_discrete (int):  number of discretization points
        midpoint (bool): take midpoints of discretization

    Returns:
        np.ndarray: discretized space, shape (dimension, n_discrete)
    """

    # check dimension of observation space (if more dimensional, interval is nested list)
    if len(np.array(interval).shape) > 1:
        return np.array(
            [
                discr_interval(interv[0], interv[1], n_discrete, midpoint)
                for interv in interval
            ]
        )
    else:
        return discr_interval(interval[0], interval[1], n_discrete, midpoint)


def discr_interval(
    lower_bound: float, upper_bound: float, n_discrete: int, midpoint: bool
) -> np.ndarray:
    """Discretize interval

    Args:
        lower_bound (float): lower bound of interval
        b (float): upper bound of interval
        n_discrete (int): number of discretization points
        midpoint (bool): if True returns midpoints of n subintervals, else a, b and n-2 points in between

    Returns:
        np.ndarray: discretized interval
    """
    if midpoint:
        return (lower_bound + (0.5 + np.arange(n_discrete)) * (upper_bound - lower_bound) / n_discrete)
    
    if (lower_bound == upper_bound) & (n_discrete > 1):
        raise ValueError("Discretized interval with n_discrete > 1 cannot have same lower and upper bound")
    if (lower_bound != upper_bound) & (n_discrete == 1):
        raise ValueError("Discretized interval with n_discrete == 1 and midpoint=False cannot have different lower and upper bounds ")
    if (lower_bound == upper_bound) & (n_discrete == 1):
        return np.array([lower_bound])
    
    return lower_bound + (np.arange(n_discrete)) * (
        upper_bound - lower_bound
    ) / (n_discrete - 1)

from itertools import product
from typing import List

import numpy as np

from soda.mechanism.mechanism import Mechanism
from soda.prior import compute_weights, marginal_prior_pdf

# -------------------------------------------------------------------------------------------------------------------- #
#                                      CLASS GAME : DISCRETIZED AUCTION GAME                                           #
# -------------------------------------------------------------------------------------------------------------------- #


class Game:
    """Game represents an approximatiion of the mechanism by discretization of the spaces

    Attributes:
        General
            bidder          List: contains all agents (str)
            set_bidder      List: contains all unique agents (model sharing)
            o_discr         Dict: discretized observation space for each agent
            a_discr         Dict: discretized action space for each agent
            n               int, number of discretization points in type space
            m               int, number of discretization points in action space
            dim_o           int, dimension of observation space
            dim_a           int, dimension of action space

    """

    def __init__(self, mechanism: Mechanism, n: int, m: int):
        """Create approximation game

        Args:
            mechanism (Mechanism): underlying mechanism
            n (int): number of discretization points for type space intervals
            m (int): number of discretization points for action space intervals
        """
        self.mechanism = mechanism
        self.n = n
        self.m = m

        self.name = mechanism.name
        self.bidder = mechanism.bidder
        self.set_bidder = mechanism.set_bidder
        self.n_bidder = mechanism.n_bidder

        self.dim_o = mechanism.dim_o
        self.dim_a = mechanism.dim_a

        # we distinguish between private, affiliated, common values ... model
        self.value_model = mechanism.value_model

        # dimension of spaces
        self.dim_o, self.dim_a = self.mechanism.dim_o, self.mechanism.dim_a

        # discrete action and observation space (and optional valuation space)
        self.o_discr = {
            i: Game.discr_spaces(mechanism.o_space[i], n, midpoint=True)
            for i in self.set_bidder
        }
        self.a_discr = {
            i: Game.discr_spaces(mechanism.a_space[i], m, midpoint=False)
            for i in self.set_bidder
        }
        if hasattr(mechanism, "v_space"):
            self.v_discr = Game.discr_spaces(mechanism.v_space, n, midpoint=True)

        # marginal prior for bidder
        self.prior = {i: self.get_prior(i) for i in self.set_bidder}

        self.weights = self.get_weights(mechanism)
        self.utility = {}

    def __repr__(self) -> str:
        return f"Game({self.name, self.n, self.m})"

    def get_utility(self) -> None:
        """Compute utility array for discretized game
        For each agent an array is stored which contains all possible combinations of
        observations/valuations and bid profiles

        Args:
            mechanism: mechanism (e.g. auction game) which has a method utility

        Raises:
            ValueError: values unknown
        """

        for i in self.set_bidder:
            index_agent = self.bidder.index(i)
            bids = self.create_all_bid_profiles()
            valuations = self.get_obs_profile_reshaped()
            shape_utilities = self.get_shape_utilities()

            self.utility[i] = (
                self.mechanism.utility(valuations, bids, index_agent)
                .transpose()
                .reshape(shape_utilities)
            )

    def create_all_bid_profiles(self) -> np.ndarray:
        """
        Create all possible action profiles (bids) for utility computation

        Returns:
            np.ndarray
        """
        if self.dim_a == 1:
            return np.array(
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
            return np.array(
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

    def get_obs_profile_reshaped(self) -> np.ndarray:
        """Due to the quasi-linear structure of the utility function, we compute the utility for one bid profile and several valuations in one step.
        This is useful for the computation of the utility array in the discretized game (to compute gradient). But to do so, we need to reformat the observations in this specific setting.

        Args:
            agent (str): agent

        Returns:
            np.ndarray: reshaped valuation of agent
        """
        # Private values (valuation = obs_agent)
        if self.value_model == "private":
            if self.dim_o == 1:
                return np.array(
                    [self.o_discr[agent].reshape(self.n, 1) for agent in self.bidder]
                )
            else:
                raise NotImplementedError

        # Common values, independent observations
        elif self.value_model == "common_independent":
            if self.dim_o == 1:
                return np.array(
                    [self.o_discr[agent].reshape(self.n, 1) for agent in self.bidder]
                    + [self.v_discr.reshape(self.n, 1)]
                )
            else:
                raise NotImplementedError

        # Common values, correlated observations (valuation = f(obs_profile))
        elif self.value_model == "common_affiliated":
            if self.dim_o == 1:
                obs = np.array(
                    np.meshgrid(*[self.o_discr[agent] for agent in self.bidder])
                )
                return obs.reshape(tuple([2] + [self.n] * self.n_bidder + [1]))
            else:
                raise NotImplementedError

    def get_shape_utilities(self) -> tuple:
        """Determine shape of utility array (depending on value model)

        Returns:
            tuple:
        """
        # Private values (valuation = obs_agent)
        if self.value_model == "private":
            return tuple(
                [self.m] * (self.dim_a * self.n_bidder) + [self.n] * self.dim_o
            )

        # Common values, independent observations
        elif self.value_model == "common_independent":
            return tuple(
                [self.m] * (self.dim_a * self.n_bidder) + [self.n] * self.dim_o
            )

        # Common values, correlated observations (valuation = f(obs_profile))
        elif self.value_model == "common_affiliated":
            return tuple(
                [self.m] * (self.dim_a * self.n_bidder)
                + [self.n] * self.dim_o * self.n_bidder
            )

    def get_prior(self, agent: str) -> np.ndarray:
        """Given prior distribution specified in mechanism, returns discretized prior
        If discretized observation space has only one entry, it corresponds to the complete information setting and the probability is equal to 1

        Args:
            agent (str): agent

        Returns:
            np.ndarray: discretized prior for agent
        """
        if self.o_discr[agent].size == 1:
            p = np.array([1])
        else:
            p = marginal_prior_pdf(self.mechanism, self.o_discr[agent], agent)
        return p / p.sum()

    def get_weights(self, mechanism):
        return compute_weights(self, mechanism)

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
                    Game.discr_interval(interv[0], interv[1], n_discrete, midpoint)
                    for interv in interval
                ]
            )
        else:
            return Game.discr_interval(interval[0], interval[1], n_discrete, midpoint)

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
        if np.isclose(lower_bound, upper_bound) ^ (n_discrete == 1):
            raise ValueError(
                "if you choose same lower and upper bound, n_discrete must be 1 (and vice versa)"
            )
        if midpoint:
            return (
                lower_bound
                + (0.5 + np.arange(n_discrete))
                * (upper_bound - lower_bound)
                / n_discrete
            )
        else:
            return lower_bound + (np.arange(n_discrete)) * (
                upper_bound - lower_bound
            ) / (n_discrete - 1)

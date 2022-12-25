from abc import abstractclassmethod
from typing import Dict, List

import numpy as np
from scipy.stats import norm, powerlaw, truncnorm, uniform

# -------------------------------------------------------------------------------------------------------------------- #
#                                         MECHANISM : CONTINUOUS AUCTION GAME                                          #
# -------------------------------------------------------------------------------------------------------------------- #


class Mechanism:
    """Mechanism defines the underlying continuous Bayesian (auction) game

    Attributes:
        General
            bidder          List: contains all agents (str)
            set_bidder      List: contains all unique agents (model sharing)
            o_space         Dict: contains limits of observation space [a,b] for agents
            a_space         Dict: contains limits of action space [a,b] for agents
            param_prior     Dict: contains parameters for prior, e.g., distribution and parameter

        Speficiations of Mechanism
            own_gradient    boolean: if true mechanism contains own function to compute gradient
            private_values  boolean:

    """

    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        param_prior: Dict,
        param_util: Dict,
    ):
        # bidder
        self.bidder = bidder
        self.n_bidder = len(bidder)
        self.set_bidder = list(set(bidder))

        # type and action space
        self.o_space = o_space
        self.a_space = a_space

        # prior
        self.prior = param_prior["distribution"]
        self.param_prior = param_prior

        # further specifications
        self.name = None
        self.param_util = param_util
        self.own_gradient = False
        self.values = "private"  # valuation depends only on own observation

    def check_bidder_symmetric(self) -> bool:
        """check if bidder have the same observation and action space

        Returns:
            bool:
        """
        o_space_identical = np.all(
            [self.o_space[0] == self.o_space[i] for i in self.set_bidder]
        )
        a_space_identical = np.all(
            [self.a_space[0] == self.a_space[i] for i in self.set_bidder]
        )
        return o_space_identical and a_space_identical

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int):
        """Compute utility according to specified mechanism

        Args:
            obs (np.ndarray): observation of agent (idx)
            bids (np.ndarray): bids of all agents
            idx (int): index of agent

        Returns:
            np.ndarry: utilities of agend (idx)
        """
        raise NotImplementedError

    def draw_values(self, n_vals: int) -> np.ndarray:
        """samples observations (and valuations) for each agent according to the prior

        Args:
            n_vals (int): number of observations per agent

        Returns:
            np.ndarray: array that contains observations for each bidder (optional: common value with highest index)
        """

        if isinstance(self.prior, dict):
            raise NotImplementedError(
                "Different priors for different bidders are not yet implemented"
            )

        if self.prior == "uniform":
            return self.draw_types_uniform(n_vals)

        elif self.prior == "gaussian":
            return self.draw_types_gaussian(n_vals)

        elif self.prior == "gaussian_trunc":
            return self.draw_types_gaussian_trunc(n_vals)

        elif self.prior == "powerlaw":
            return self.draw_types_powerlaw(n_vals)

        elif self.prior == "affiliated_values":
            w = uniform.rvs(loc=0, scale=1, size=(3, n_vals))
            return np.array([w[0] + w[2], w[1] + w[2]])

        elif self.prior == "common_value":
            w = uniform.rvs(loc=0, scale=1, size=(self.n_bidder + 1, n_vals))
            return np.array([2 * w[i] * w[3] for i in range(self.n_bidder)] + [w[3]])

        else:
            raise ValueError('prior "' + self.prior + '" not implement')

    def draw_types_uniform(self, n_types: int) -> np.ndarray:
        """draw types according to uniform distribution
        Correlation (Bernoulli weights model) only implemented for 2 bidder

        Args:
            n_types (int): number of types per bidder

        Returns:
            np.ndarray: array with types for each bidder
        """
        if "corr" not in self.param_prior:
            return np.array(
                [
                    uniform.rvs(
                        loc=self.o_space[self.bidder[i]][0],
                        scale=self.o_space[self.bidder[i]][-1]
                        - self.o_space[self.bidder[i]][0],
                        size=n_types,
                    )
                    for i in range(self.n_bidder)
                ]
            )
        elif (
            ("corr" in self.param_prior)
            & (self.n_bidder == 2)
            & self.check_bidder_symmetric()
        ):
            gamma = self.param_prior["corr"]
            w = np.random.uniform(size=n_types)
            u = uniform.rvs(
                loc=self.o_space[self.bidder[0]][0],
                scale=self.o_space[self.bidder[0]][-1]
                - self.o_space[self.bidder[0]][0],
                size=(3, n_types),
            )
            return np.array(
                [
                    np.where(w < gamma, 1, 0) * u[2] + np.where(w < gamma, 0, 1) * u[0],
                    np.where(w < gamma, 1, 0) * u[2] + np.where(w < gamma, 0, 1) * u[1],
                ]
            )
        else:
            raise NotImplementedError(
                "Correlation with Uniform Prior only implemented for 2 bidder"
            )

    def draw_types_gaussian(self, n_types: int) -> np.ndarray:
        """draw types according to gaussian distribution

        Args:
            n_types (int): number of types per bidder

        Returns:
            np.ndarray: array with types for each bidder
        """
        if ("mu" in self.param_prior) & ("sigma" in self.param_prior):
            mu, sigma = self.param_prior["mu"], self.param_prior["sigma"]
        else:
            raise ValueError("'mu' and 'sigma' must be specified in param_prior")

        return np.array(
            [
                norm.rvs(
                    loc=mu,
                    scale=sigma,
                    size=n_types,
                )
                for i in range(self.n_bidder)
            ]
        )

    def draw_types_gaussian_trunc(self, n_types: int) -> np.ndarray:
        """draw types according to truncated gaussian distribution

        Args:
            n_types (int): number of types per bidder

        Returns:
            np.ndarray: array with types for each bidder
        """
        if ("mu" in self.param_prior) & ("sigma" in self.param_prior):
            mu, sigma = self.param_prior["mu"], self.param_prior["sigma"]
        else:
            raise ValueError("'mu' and 'sigma' must be specified in param_prior")

        return np.array(
            [
                truncnorm.rvs(
                    a=(self.o_space[i][0] - mu) / sigma,
                    b=(self.o_space[i][1] - mu) / sigma,
                    loc=mu,
                    scale=sigma,
                    size=n_types,
                )
                for i in self.bidder
            ]
        )

    def draw_types_powerlaw(self, n_types: int) -> np.ndarray:
        """draw types according to powerlaw distribution

        Args:
            n_types (int): number of types per bidder

        Returns:
            np.ndarray: array with types for each bidder
        """
        if "power" in self.param_prior:
            power = self.param_prior["power"]
        else:
            raise ValueError(
                "'power' must be specified in param_prior for powerlaw distribution"
            )

        return np.array(
            [
                uniform.rvs(
                    a=power,
                    loc=self.o_space[self.bidder[i]][0],
                    scale=self.o_space[self.bidder[i]][-1]
                    - self.o_space[self.bidder[i]][0],
                    size=n_types,
                )
                for i in range(self.n_bidder)
            ]
        )

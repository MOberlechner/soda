from typing import Dict, List

import numpy as np
from scipy.stats import norm, powerlaw, uniform

# -------------------------------------------------------------------------------------------------------------------- #
#                                   CLASS MECHANISM : CONTINUOUS AUCTION GAME                                          #
# -------------------------------------------------------------------------------------------------------------------- #


class Mechanism:
    """Mechanism defines the underlying continuous Bayesian (auction) game

    Attributes:
        General
            bidder (List) : contains all agents (str)
            set_bidder (List): contains all unique agents (model sharing)
            o_space (Dict): contains limits of observation space [a,b] for agents
            a_space (Dict): contains limits of action space [a,b] for agents
            param_prior (Dict): contains parameters for prior, e.g., distribution and parameter

        Speficiations of Mechanism
            own_gradient (boolean): if true mechanism contains own function to compute gradient
            private_values (boolean): u

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
        self.param_util = param_util
        self.own_gradient = False
        self.private_values = True  # valuation depends only on own observation

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
            # independent prior
            if "corr" not in self.param_prior:
                return np.array(
                    [
                        uniform.rvs(
                            loc=self.o_space[self.bidder[i]][0],
                            scale=self.o_space[self.bidder[i]][-1]
                            - self.o_space[self.bidder[i]][0],
                            size=n_vals,
                        )
                        for i in range(self.n_bidder)
                    ]
                )
            # correlated prior with Bernoulli Paramater (according to Ausubel & Baranov 2020)
            else:
                if self.n_bidder == 2 and "corr" in self.param_prior:
                    gamma = self.param_prior["corr"]
                    w = np.random.uniform(size=n_vals)
                    u = uniform.rvs(
                        loc=self.o_space[self.bidder[0]][0],
                        scale=self.o_space[self.bidder[0]][-1]
                        - self.o_space[self.bidder[0]][0],
                        size=(3, n_vals),
                    )
                    return np.array(
                        [
                            np.where(w < gamma, 1, 0) * u[2]
                            + np.where(w < gamma, 0, 1) * u[0],
                            np.where(w < gamma, 1, 0) * u[2]
                            + np.where(w < gamma, 0, 1) * u[1],
                        ]
                    )
                else:
                    raise NotImplementedError

        elif self.prior == "gaussian":
            return np.array(
                [
                    norm.rvs(
                        loc=self.param_prior["mu"],
                        scale=self.param_prior["sigma"],
                        size=n_vals,
                    )
                    for i in range(self.n_bidder)
                ]
            )

        elif self.prior == "powerlaw":
            power = self.param_prior["power"]
            return np.array(
                [
                    uniform.rvs(
                        a=power,
                        loc=self.o_space[self.bidder[i]][0],
                        scale=self.o_space[self.bidder[i]][-1]
                        - self.o_space[self.bidder[i]][0],
                        size=n_vals,
                    )
                    for i in range(self.n_bidder)
                ]
            )

        elif self.prior == "affiliated_values":
            w = uniform.rvs(loc=0, scale=1, size=(3, n_vals))
            return np.array([w[0] + w[2], w[1] + w[2]])

        elif self.prior == "common_values":
            w = uniform.rvs(loc=0, scale=1, size=(self.n_bidder + 1, n_vals))
            return np.array(
                [2 * w[i] * w[3] for i in range(self.n_bidder)] + [w[self.n_bidder]]
            )

        else:
            raise ValueError('prior "' + self.prior + '" not implement')

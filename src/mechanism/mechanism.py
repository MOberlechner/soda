from typing import Dict, List

import numpy as np
from scipy.stats import norm, powerlaw, uniform

# -------------------------------------------------------------------------------------------------------------------- #
#                                   CLASS MECHANISM : CONTINUOUS AUCTION GAME                                          #
# -------------------------------------------------------------------------------------------------------------------- #


class Mechanism:
    """Mechanism

    A Mechanism object defines the continuous auction game. Are relevant properties of the incomplete information
    game G = (I, O, A, u , F) as well as some basic methods as draw_valuations, should be contained.
    """

    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        param_prior: Dict[str, str],
    ):
        """

        Parameters
        ----------
        bidder : list, of bidders, e.g. ['1', '2']
        o_space : dict, that contains lists with lower and upper bounds of intervals ofr each bidder
        a_space : dict, that contains lists with lower and uppejupr bounds of intervals ofr each bidder
        param_prior : dict, that contains parameters in forms of dictionaries for each bidder
        """

        self.bidder = bidder
        self.n_bidder = len(bidder)
        self.set_bidder = list(set(bidder))
        self.o_space = o_space
        self.a_space = a_space
        self.prior = param_prior["distribution"]
        self.param_prior = param_prior
        self.own_gradient = False

    def draw_values(self, n_vals: int):
        """Valuations are drawn according to the given prior. If agents have different kind of priors, i.e., not only
        different intervals, than this has to be implemented separately in the corresponding mechanism

        Parameters
        ----------
        n_vals :

        Returns
        -------

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
                    raise NotImplementedError()

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

        else:
            raise ValueError('prior "' + self.prior + '" not implement')

    def prior_pdf(self, obs: np.ndarray, agent: str):

        if self.prior == "uniform":
            return uniform.pdf(
                obs,
                loc=self.o_space[agent][0],
                scale=self.o_space[agent][-1] - self.o_space[agent][0],
            )

        if self.prior == "uniform_bi":
            eta = 0.9
            return eta * uniform.pdf(
                obs,
                loc=self.o_space[agent][0],
                scale=self.o_space[agent][-1] - self.o_space[agent][0],
            ) + (1 - eta) * uniform.pdf(
                obs,
                loc=0.4 * (self.o_space[agent][0] + self.o_space[agent][-1]),
                scale=0.2 * (self.o_space[agent][-1] - self.o_space[agent][0]),
            )

        elif self.prior == "gaussian":
            return norm.pdf(
                obs, loc=self.param_prior["mu"], scale=self.param_prior["sigma"]
            )

        elif self.prior == "gaussian_bimodal":
            eta = 0.5
            return eta * norm.pdf(obs, loc=0.25, scale=0.1) + (1 - eta) * norm.pdf(
                obs, loc=0.75, scale=0.1
            )

        elif self.prior == "powerlaw":
            power = self.param_prior["power"][agent]
            return powerlaw.pdf(
                obs,
                a=power,
                loc=self.o_space[agent][0],
                scale=self.o_space[agent[0]][-1] - self.o_space[agent][0],
            )

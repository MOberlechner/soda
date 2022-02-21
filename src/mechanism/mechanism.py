from typing import Dict, List

import numpy as np
from scipy.stats import powerlaw, uniform

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
        param_prior: Dict,
    ):

        self.bidder = bidder
        self.n_bidder = len(bidder)
        self.set_bidder = list(set(bidder))
        self.o_space = o_space
        self.a_space = a_space
        self.prior = param_prior["distribution"]
        self.param_prior = param_prior

    def draw_values(self, n_vals: int):

        if self.prior == "uniform":
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

        elif self.prior == "gaussian":
            pass

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
                scale=self.o_space[agent[0]][-1] - self.o_space[agent][0],
            )

        elif self.prior == "powerlaw":
            power = self.param_prior["power"]
            return powerlaw.pdf(
                obs,
                a=power,
                loc=self.o_space[agent][0],
                scale=self.o_space[agent[0]][-1] - self.o_space[agent][0],
            )

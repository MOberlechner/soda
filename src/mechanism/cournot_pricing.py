from typing import Dict, List

import numpy as np
from scipy.special import binom

from .mechanism import Mechanism

# -------------------------------------------------------------------------------------------------------------------- #
#                                           COURNOT PRICING COMPETITION                                                #
# -------------------------------------------------------------------------------------------------------------------- #


class CournotPricing(Mechanism):
    """Cournot pricing competition

    Parameter Mechanism
        bidder, o_space, a_space - standard input for all mechanism (see class Mechanism)

    Parameter Prior (param_prior)
        distribution    str: "uniform" or "gaussian"; "corr"

    Parameter Utility (param_util)
        demand          str: "linear"
          - slope       int: specifies the slope of the linear demand function, defaults to 1 -> only required for linear demand
          - intercept   int: specifies the intercept of the linear demand function, defaults to 1 -> only required for linear demand

    """

    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        param_prior: Dict[str, str],
        param_util: Dict,
    ):
        super().__init__(bidder, o_space, a_space, param_prior, param_util)
        self.name = "cournot_pricing"

        # check input
        if "demand" not in self.param_util:
            raise ValueError("specify demand function")

        self.demand = param_util["demand"]

        if self.demand == "linear":
            if "slope" not in self.param_util:
                raise ValueError("specify slope of linear demand function")
            elif "intercept" not in self.param_util:
                raise ValueError("specify intercept of linear demand function")

            self.slope = param_util["slope"]
            self.intercept = param_util["intercept"]

            if self.slope < 1:
                raise ValueError("slope must be at least 1")
            elif self.intercept <= 0:
                raise ValueError("intercept must be greater than 0")

        else:
            raise ValueError("demand function '" + str(self.demand) + "' not available")

        if self.prior in ["affiliated_values", "common_value"]:
            raise NotImplementedError

        """
        # TODO: probably not required -> use own gradient
        if (len(self.set_bidder) == 1) and (self.demand == "linear") & (
            self.prior not in ["affiliated_values", "common_value"]
        ) & ("corr" not in self.param_prior):
            self.own_gradient = True
            print("own gradient")
        """

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int) -> None:
        """
        Payoff function for Cournot pricing competition

        Parameters
        ----------
        obs : observation/marginal cost of bidder (idx)
        bids : array with bid profiles
        idx : index of bidder to consider

        Returns
        -------
        np.ndarray : payoff vector for agent idx

        """

        # test input
        if bids.shape[0] != self.n_bidder:
            raise ValueError("wrong format of bids")
        elif idx >= self.n_bidder:
            raise ValueError("bidder with index " + str(idx) + " not available")

        # if True: we want each outcome for every observation
        if obs.shape != bids[idx].shape:
            obs = obs.reshape(len(obs), 1)

        # determine price
        market_quantity = bids.sum(axis=0)
        price = (1 / self.slope) * (self.intercept - market_quantity)

        # determine payoff
        if self.demand == "linear":
            return (price - obs) * bids[idx]

        else:
            raise ValueError("demand function '" + self.demand + "' not available")

    def get_bne(self, agent: str, obs: np.ndarray):
        """
        Returns BNE for some predefined settings

        Parameters
        ----------
        agent : specficies bidder (important in asymmetric settings)
        obs :  observation/marginal cost of agent

        Returns
        -------
        np.ndarray : bids to corresponding observation

        """

        if self.prior == "uniform":
            if (self.demand == "linear") & np.all(
                [
                    self.o_space[i] == [0, self.slope / self.intercept]
                    for i in self.set_bidder
                ]
            ):
                BNE_vfunc = np.vectorize(self.BNE_func)

                return BNE_vfunc(obs)

    def BNE_func(self, obs: np.ndarray):
        coeff = (
            2
            * (
                np.sqrt(
                    self.intercept * self.slope * (self.n_bidder - 1)
                    + np.square(self.slope)
                )
                - self.slope
            )
            / (self.n_bidder - 1)
        )
        if obs >= coeff / self.slope:
            return 0
        else:
            return (coeff - self.slope * obs) / 2

    """
    TODO Laura: verify whether an own gradient can be implemented (cf. function compute_gradient(...))
    """

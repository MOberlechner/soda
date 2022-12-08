from typing import Dict, List

import numpy as np
from scipy.special import binom

from .mechanism import Mechanism

# -------------------------------------------------------------------------------------------------------------------- #
#                                          BERTRAND PRICING COMPETITION                                                #
# -------------------------------------------------------------------------------------------------------------------- #


class BertrandPricing(Mechanism):
    """Bertrand pricing competition

    Parameter Mechanism
        bidder, o_space, a_space - standard input for all mechanism (see class Mechanism)

    Parameter Prior (param_prior)
        distribution    str: "uniform" or "gaussian"; "corr"

    Parameter Utility (param_util)
        tie_breaking    str: "random" or "lose"

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
        self.name = "bertrand_pricing"

        # check input
        if "tie_breaking" not in self.param_util:
            raise ValueError("specify tiebreaking rule")
        if "demand" not in self.param_util:
            raise ValueError("specify demand function")

        self.tie_breaking = param_util["tie_breaking"]
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

        if self.prior in ["affiliated_values", "common_value"]:
            raise NotImplementedError

        else:
            raise ValueError("demand function '" + str(self.demand) + "' not available")

        # use own gradient
        if (len(self.set_bidder) == 1) and (self.demand == "linear") & (
            self.prior not in ["affiliated_values", "common_value"]
        ) & ("corr" not in self.param_prior):
            self.own_gradient = True
            print("own gradient")

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int) -> None:
        """
        Payoff function for Bertrand pricing competition

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

        # determine allocation
        if self.tie_breaking == "random":
            win = np.where(bids[idx] <= np.delete(bids, idx, 0).min(axis=0), 1, 0)
            num_winner = (bids.min(axis=0) == bids).sum(axis=0)

        elif self.tie_breaking == "lose":
            win = np.where(bids[idx] < np.delete(bids, idx, 0).min(axis=0), 1, 0)
            num_winner = np.ones(win.shape)
        else:
            raise ValueError(
                'Tie-breaking rule "' + self.param_util["tie_breaking"] + '" unknown'
            )

        # determine payoff
        if self.demand == "linear":
            payoff = (self.intercept - self.slope * bids[idx]) * (bids[idx] - obs)

            return 1 / num_winner * win * payoff

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
                return (
                    self.slope / (self.intercept * (1 + self.n_bidder))
                    + ((self.n_bidder) / (1 + self.n_bidder)) * obs
                )

    def compute_gradient(self, game, strategies, agent: str):
        """Simplified computation of gradient for i.i.d. bidders

        Parameters
        ----------
        strategies :
        game :
        agent :

        Returns
        -------

        """
        pdf = strategies[agent].x.sum(axis=0)
        cdf = 1 - np.cumsum(pdf)
        exp_win = cdf ** (self.n_bidder - 1)

        if self.tie_breaking == "lose":
            pass
        elif self.tie_breaking == "random":
            exp_win += sum(
                [
                    binom(self.n_bidder - 1, i)
                    * cdf ** (self.n_bidder - i - 1)
                    / (i + 1)
                    * pdf**i
                    for i in range(1, self.n_bidder)
                ]
            )
        else:
            raise ValueError('Tie-breaking rule "{}" unknown'.format(self.tie_breaking))

        obs_grid = (
            np.ones((strategies[agent].m, strategies[agent].n))
            * strategies[agent].o_discr
        ).T
        bid_grid = (
            np.ones((strategies[agent].n, strategies[agent].m))
            * strategies[agent].a_discr
        )

        if self.demand == "linear":
            payoff = (bid_grid - obs_grid) * (
                (np.ones((strategies[agent].n, strategies[agent].m)) * self.slope)
                - (self.intercept * bid_grid)
            )

        else:
            raise ValueError("demand function '" + self.demand + "' not available")

        return exp_win * payoff

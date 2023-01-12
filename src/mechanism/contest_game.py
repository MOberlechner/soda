from typing import Dict, List

import numpy as np

from .mechanism import Mechanism

# -------------------------------------------------------------------------------------------------------------------- #
#                                                   CONTEST GAME                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


class ContestGame(Mechanism):
    """Contest-Game

    Parameter Mechanism
        bidder, o_space, a_space - standard input for all mechanism (see class Mechanism)

    Parameter Prior (param_prior)
        distribution

    Parameter Utility (param_util)
        csf         str: contest success function (ratio_form or difference form), determines tie
                    probability of winning

        csf_param   float: parameter for contest success function

        type        str: agent's type can be interpreted as "valuation" or "cost"

    """

    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        param_prior: Dict,
        param_util: Dict,
    ):
        super().__init__(bidder, o_space, a_space, param_prior, param_util)
        self.name = "contest_game"

        self.check_param()

        self.csf = param_util["csf"]
        self.param_csf = param_util["csf_param"]
        self.type = param_util["type"]

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int) -> np.ndarray:
        """Utility function for contest game

        Args:
            obs (np.ndarray): valuation/cost parameter (type)
            bids (np.ndarray): bid profiles
            idx (int): agent

        Returns:
            np.ndarray: payoff vector for agent
        """

        self.test_input_utility(obs, bids, idx)
        valuation = self.get_valuation(obs, bids, idx)

        allocation = self.get_allocation(bids, idx)
        payoff = self.get_payoff(valuation, allocation, bids[idx])
        return payoff

    def get_payoff(
        self, valuation: np.ndarray, allocation: np.ndarray, bid: np.ndarray
    ) -> np.ndarray:
        """compute payoff given type and allocation

        Args:
            valuation (np.ndarray): type (valuation or cost parameter) for agent
            allocation (np.ndarray): winning probability for agent
            bid (np.ndarray): bid of agent

        Returns:
            np.ndarray: payoff
        """
        if self.type == "cost":
            return allocation - valuation * bid
        elif self.type == "value":
            return valuation * allocation - bid
        else:
            raise ValueError(
                "Type unknown: choose either cost or value for type in param_util"
            )

    def get_allocation(self, bids: np.ndarray, idx: int) -> np.ndarray:
        """compute allocation (probability) given action profiles for agent i

        Args:
            bids (np.ndarray): action profiles
            idx (int): index of agent we consider

        Returns:
            np.ndarray: allocation vector for agent idx
        """
        if self.csf == "ratio_form":
            r = self.param_csf
            return (bids[idx] ** r + np.where(bids.sum(axis=0) == 0, 1, 0)) / (
                (bids**r).sum(axis=0)
                + np.where(bids.sum(axis=0) == 0, self.n_bidder, 0)
            )

        elif self.csf == "difference_form":
            mu = self.param_csf
            return np.exp(mu * bids[idx]) / np.exp(mu * bids).sum(axis=0)

        else:
            raise NotImplementedError(f"contest succes function {self.csf} unknown")

    def check_param(self):
        """
        Check if input paremter are sufficient to define mechanism
        """
        if "csf" not in self.param_util:
            raise ValueError(
                "specify contest succes function (csf) in param_util - csv: ratio_form or difference_form"
            )

        if "csf_param" not in self.param_util:
            raise ValueError(
                "specify parameter csf_param for contest succes function (csf) in param_util"
            )

        if "type" not in self.param_util:
            raise ValueError("Specify type in param_util - type: cost or value")

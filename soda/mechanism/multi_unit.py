from typing import Dict, List

import numpy as np
from scipy.special import binom
from scipy.stats import rankdata

from .mechanism import Mechanism

# -------------------------------------------------------------------------------------------------------------------- #
#                                              MULTI-UNIT AUCTION                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


class MultiUnitAuction(Mechanism):
    """Multi-Unit Auction

    Parameter Mechanism
        bidder, o_space, a_space - standard input for all mechanism (see class Mechanism)


    Parameter Prior (param_prior)
        distribution    str: "affiliated_values" and "common_value" are available


    Parameter Utility (param_util)
        tiebreaking     str: specifies tiebreaking rule: "random" (default), "lose"
        payment_rule    str: choose betweem "uniform"
        number_items    int: number of items

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
        self.name = "multi_unit"

        self.check_param()
        self.payment_rule = param_util["payment_rule"]
        self.tie_breaking = param_util["tie_breaking"]
        self.number_items = param_util["number_items"]

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int) -> None:
        """
        Payoff function for first price sealed bid auctons

        Parameters
        ----------
        obs : observation/valuation of bidder (idx)
        bids : array with bid profiles
        idx : index of bidder to consider

        Returns
        -------
        np.ndarray : payoff vector for agent idx

        """
        self.test_input_utility(obs, bids, idx)
        valuations = self.get_valuation(obs, bids, idx)
        allocation = self.get_allocation(bids)

        payments = self.get_payment(bids, allocation, idx)
        payoff = allocation.sum(axis=1)[idx] * valuations - (
            allocation[idx] * payments
        ).sum(axis=0)

        return payoff

    def get_allocation(self, bids: np.ndarray) -> np.ndarray:
        """compute allocation given action profiles for all bidder

        Args:
            bids (np.ndarray): action profiles, shape (n_bidder x n_items x n_bids)
            idx (int): index of agent we consider

        Returns:
            np.ndarray: array (shape of bids) indicating which bid won
        """

        if self.tie_breaking == "lose":
            rank = rankdata(
                (-bids).reshape(self.n_bidder * self.number_items, -1),
                method="max",
                axis=0,
            ).reshape(bids.shape)
        else:
            raise NotImplementedError(
                f"tie-breaking rule {self.tie_breaking} is not implemented"
            )

        allocation = np.zeros_like(bids)
        allocation[rank <= self.number_items] = 1

        return allocation

    def get_payment(
        self, bids: np.ndarray, allocation: np.ndarray, idx: int
    ) -> np.ndarray:
        """compute payment (assuming bidder idx wins)

        Args:
            bids (np.ndarray): action profiles
            idx (int): index of agent we consider

        Returns:
            np.ndarray: payment vector
        """
        # pay own bid
        if self.payment_rule == "first_price":
            payments = np.zeros_like(bids[idx])
            payments[np.array(allocation, dtype=bool)[idx]] = bids[idx][
                np.array(allocation, dtype=bool)[idx]
            ]
            payments = payments.sum(axis=0)

        # highest losing bid
        elif self.payment_rule == "uniform_price":
            payments = np.zeros_like(bids)
            payments[~np.array(allocation, dtype=bool)] = bids[
                ~np.array(allocation, dtype=bool)
            ]
            payments = payments.reshape(self.n_bidder * self.number_items, -1).max(
                axis=0
            )

        else:
            raise ValueError("payment rule " + self.payment_rule + " not available")

        return payments

    def get_valuation(self, obs: np.ndarray, bids: np.ndarray, idx: int) -> np.ndarray:
        """determine valuations (potentially from observations, might be equal for private value model)
        and reformat vector depending on the use case:
            - one valuation for each action profile (no reformatting), needed for simulation
            - all valuations for each action profule (reformatting), needed for gradient computation (game.py)

        Args:
            obs (np.ndarray): observation of agent (idx)
            bids (np.ndarray): bids of all agents
            idx (int): index of agent

        Returns:
            np.ndarray: observations, possibly reformated
        """
        if (self.value_model == "private") or (self.value_model == "common"):
            if obs.shape != bids[idx].shape:
                valuations = obs.reshape(len(obs), 1)
            else:
                valuations = obs
        elif self.value_model == "affiliated":
            if obs[idx].shape != bids[idx].shape:
                valuations = 0.5 * (
                    obs.reshape(len(obs), 1) + obs.reshape(1, len(obs))
                ).reshape(len(obs), len(obs), 1)
            else:
                valuations = obs.mean(axis=0)
        else:
            raise NotImplementedError(f"value model {self.value_model} unknown")
        return valuations

    def get_bne(self, agent: str, obs: np.ndarray):
        """
        Returns BNE for some predefined settings

        Parameters
        ----------
        agent : specficies bidder (important in asymmetric settings)
        obs :  observation/valuation of agent

        Returns
        -------
        np.ndarray : bids to corresponding observation

        """
        pass

    def check_param(self):
        """Check if input paremter are sufficient to define mechanism"""
        if "tie_breaking" not in self.param_util:
            raise ValueError("specify tiebreaking rule")
        if "payment_rule" not in self.param_util:
            raise ValueError("specify payment rule")
        if "number_items" not in self.param_util:
            raise ValueError("specify number of items")

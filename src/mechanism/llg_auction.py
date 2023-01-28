from typing import Dict, List

import numpy as np
from scipy.stats import uniform

from .mechanism import Mechanism

# -------------------------------------------------------------------------------------------------------------------- #
#                                          COMBINATORAL AUCTION - LLG                                                  #
# -------------------------------------------------------------------------------------------------------------------- #


class LLGAuction(Mechanism):
    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        param_prior: Dict,
        param_util: Dict,
    ):

        super().__init__(bidder, o_space, a_space, param_prior, param_util)
        self.name = "llg_auction"

        self.check_param()
        self.payment_rule = param_util["payment_rule"]
        self.tie_breaking = param_util["tie_breaking"]
        self.gamma = param_prior["corr"]

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int):
        """Payoff function for first price sealed bid auctons

        Parameters
        ----------
        obs : observation/valuation of bidder (idx)
        bids : array with bid profiles
        idx : index of bidder to consider

        Returns
        -------
        utility for the LLG-Auction under different bidder-optimal core-selecting rules
        """

        self.test_input_utility(obs, bids, idx)
        valuation = self.get_valuation(obs, bids, idx)

        allocation = self.get_allocation(bids, idx)
        payment = self.get_payment(bids, allocation, idx)

        return allocation * valuation - payment

    def get_allocation(self, bids: np.ndarray, idx: int) -> tuple:
        """compute allocation given action profiles

        Args:
            bids (np.ndarray): action profiles
            idx (int): index of agent we consider

        Returns:
            tuple: 2x np.ndarray, allocation vector for agent idx, number of ties
        """
        # global bidder
        if idx == 2:
            is_winner = np.where(bids[2] > bids[:2].sum(axis=0), 1, 0)
            param_tie = {"random": 0.5, "local": 0.0, "lose": 0.0}[self.tie_breaking]
            tie = np.where(bids[2] == bids[:2].sum(axis=0), param_tie, 0)
        # local bidder
        else:
            is_winner = np.where(bids[2] < bids[:2].sum(axis=0), 1, 0)
            param_tie = {"random": 0.5, "local": 1.0, "lose": 0.0}[self.tie_breaking]
            tie = np.where(bids[2] == bids[:2].sum(axis=0), param_tie, 0)
        allocation = is_winner + tie
        return allocation

    def get_payment(self, bids: np.ndarray, allocation: np.ndarray, idx: int):
        """compute payment (assuming bidder idx wins) for different payment rules

        Args:
            bids (np.ndarray): action profiles
            alloaction (np.ndarray): allocation vector for agent idx
            idx (int): index of agent we consider

        Returns:
            np.ndarray: payment vector
        """
        # global bidder
        if idx == 2:
            payment = bids[:2].sum(axis=0)

        # local bidder
        else:
            # Nearest-Zero (Proxy Rule)
            if self.payment_rule == "NZ":
                case_a = bids[2] <= 2 * bids[:2].min(axis=0)
                payment = case_a * 0.5 * bids[2] + (1 - case_a) * np.where(
                    bids[idx] == bids[:2].min(axis=0),
                    bids[idx],
                    bids[2] - bids[:2].min(axis=0),
                )

            # Nearest-VCG Rule
            elif self.payment_rule == "NVCG":
                vcg_payments = {
                    0: -bids[1] + np.maximum(bids[1], bids[2]),
                    1: -bids[0] + np.maximum(bids[0], bids[2]),
                }
                delta = 0.5 * (bids[2] - vcg_payments[0] - vcg_payments[1])
                payment = vcg_payments[idx] + delta

            # Nearest-Bid Rule
            elif self.payment_rule == "NB":
                case_a = bids[2] <= bids[:2].max(axis=0) - bids[:2].min(axis=0)
                delta = 0.5 * (bids[:2].sum(axis=0) - bids[2])
                payment = case_a * np.where(
                    bids[idx] == bids[:2].max(axis=0), bids[2], 0
                ) + (1 - case_a) * (bids[idx] - delta)

            else:
                raise ValueError("payment rule unknown")

        return payment * np.where(allocation > 0, 1.0, 0.0)

    def sample_types_uniform(self, n_types: int) -> np.ndarray:
        """owerwrite method from mechanism class:
        draw types according to uniform distribution
        Correlation (Bernoulli weights model) implemented for local bidders

        Args:
            n_types (int): number of types per bidder

        Returns:
            np.ndarray: array with types for each bidder
        """

        # independent prior
        if self.gamma > 0:
            w = uniform.rvs(loc=0, scale=1, size=n_types)
            u = uniform.rvs(
                loc=self.o_space[self.bidder[0]][0],
                scale=self.o_space[self.bidder[0]][-1]
                - self.o_space[self.bidder[0]][0],
                size=(3, n_types),
            )
            return np.array(
                [
                    # local 1
                    np.where(w < self.gamma, 1, 0) * u[2]
                    + np.where(w < self.gamma, 0, 1) * u[0],
                    # local 2
                    np.where(w < self.gamma, 1, 0) * u[2]
                    + np.where(w < self.gamma, 0, 1) * u[1],
                    # global
                    uniform.rvs(
                        loc=self.o_space["G"][0],
                        scale=self.o_space["G"][1] - self.o_space["G"][0],
                        size=n_types,
                    ),
                ]
            )
        else:
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

    def get_bne(self, agent: str, obs: np.ndarray) -> np.ndarray:
        """
        Returns BNE for different payment rulse (assuming observations and actions for local bidders are [0,1])

        Args:
            agent (str): specifies bidder, either local L or global G
            obs (np.ndarray): _description_

        Returns:
            np.ndarray: equilibrium bid for each observation
        """
        if agent == "G":
            return obs
        elif agent == "L":
            if self.payment_rule == "NVCG":
                if self.gamma == 1:
                    return 2 / 3 * obs
                else:
                    obs_star = (3 - np.sqrt(9 - (1 - self.gamma) ** 2)) / (
                        1 - self.gamma
                    )
                    return np.maximum(
                        np.zeros(len(obs)), 2 / (2 + self.gamma) * (obs - obs_star)
                    )

            elif self.payment_rule == "NZ":
                return np.maximum(
                    np.zeros(len(obs)),
                    np.log(self.gamma + (1 - self.gamma) * obs) / (1 - self.gamma) + 1,
                )

            elif self.payment_rule == "NB":
                if self.gamma == 1:
                    return 1 / 2 * obs
                else:
                    return (
                        1
                        / (1 - self.gamma)
                        * (np.log(2) - np.log(2 - (1 - self.gamma) * obs))
                    )

            else:
                raise ValueError("BNE not available: payment rule unknown")
        else:
            raise ValueError("BNE only for local (L) or global (G) bidder.")

    def check_param(self):
        """
        Check if input paremter are sufficient to define mechanism
        """
        if "tie_breaking" not in self.param_util:
            raise ValueError("specify tiebreaking rule")
        elif self.param_util["tie_breaking"] not in ["local", "lose", "random"]:
            raise ValueError("tiebreaking rule is not available")

        if "payment_rule" not in self.param_util:
            raise ValueError("specify payment rule")
        elif self.param_util["payment_rule"] not in ["NVCG", "NZ", "NB"]:
            raise ValueError("payment rule is not available")

        if self.bidder[2] != "G":
            raise ValueError('choose either ["L","L","G"] or  ["L1","L2","G"] ')

        if "corr" not in self.param_prior:
            self.param_prior["corr"] = 0.0
            print(
                "No correlation for LLG Auction defined. We assume no correlation (corr=0)."
            )

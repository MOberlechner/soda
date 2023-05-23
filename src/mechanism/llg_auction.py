from typing import Dict, List

import numpy as np
from scipy.stats import uniform

from .mechanism import Mechanism

# -------------------------------------------------------------------------------------------------------------------- #
#                                          COMBINATORAL AUCTION - LLG                                                  #
# -------------------------------------------------------------------------------------------------------------------- #


class LLGAuction(Mechanism):
    """Combinatorial Auction: LLG-Model

    Parameter Mechanism
        bidder, o_space, a_space - standard input for all mechanism (see class Mechanism)


    Parameter Prior (param_prior)
        distribution
        corr            float, correlation of observations/valuation of local bidders


    Parameter Utility (param_util)
        tiebreaking     str: specifies tiebreaking rule: random, lose or local
        payment_rule    str, nearest_vcg, nearest_bid, nearest_zero or first_price

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
        self.name = "llg_auction"

        self.check_param()
        self.payment_rule = param_util["payment_rule"]
        self.tie_breaking = param_util["tie_breaking"]
        self.gamma = param_prior["corr"]

    def utility(
        self, obs_profile: np.ndarray, bids_profile: np.ndarray, index_agent: int
    ) -> np.ndarray:
        """Compute utility for LLG Auction

        Args:
            obs_profile (np.ndarray): observations of all agents
            bids_profile (np.ndarray): bids of all agents
            index_agent (int): index of agent

        Returns:
            np.ndarry: utilities of agent (with index index_agent)
        """

        self.test_input_utility(obs_profile, bids_profile, index_agent)
        valuation = self.get_valuation(obs_profile, index_agent)

        allocation = self.get_allocation(bids_profile, index_agent)
        payment = self.get_payment(bids_profile, allocation, index_agent)

        return allocation * (valuation - payment)

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
        """compute payment for different payment rules
        we do not consider tie-breaking rules, if allocation > 0, full payment is computed

        Args:
            bids (np.ndarray): action profiles
            alloaction (np.ndarray): allocation vector for agent idx
            idx (int): index of agent we consider

        Returns:
            np.ndarray: payment vector
        """

        if self.payment_rule == "nearest_zero":
            if idx == 2:  # global bidder
                payment = bids[:2].sum(axis=0)
            else:  # local bidder
                case_a = bids[2] <= 2 * bids[:2].min(axis=0)
                potential_payment = case_a * 0.5 * bids[2] + (1 - case_a) * np.where(
                    bids[idx] == bids[:2].min(axis=0),
                    bids[idx],
                    bids[2] - bids[:2].min(axis=0),
                )

        elif self.payment_rule == "nearest_vcg":
            if idx == 2:  # global bidder
                potential_payment = bids[:2].sum(axis=0)
            else:  # local bidders
                vcg_payments = {
                    0: -bids[1] + np.maximum(bids[1], bids[2]),
                    1: -bids[0] + np.maximum(bids[0], bids[2]),
                }
                delta = 0.5 * (bids[2] - vcg_payments[0] - vcg_payments[1])
                potential_payment = vcg_payments[idx] + delta

        elif self.payment_rule == "nearest_bid":
            if idx == 2:  # global bidder
                potential_payment = bids[:2].sum(axis=0)
            else:  # local bidders
                case_a = bids[2] <= bids[:2].max(axis=0) - bids[:2].min(axis=0)
                delta = 0.5 * (bids[:2].sum(axis=0) - bids[2])
                potential_payment = case_a * np.where(
                    bids[idx] == bids[:2].max(axis=0), bids[2], 0
                ) + (1 - case_a) * (bids[idx] - delta)

        elif self.payment_rule == "first_price":
            potential_payment = bids[idx]

        else:
            raise ValueError("payment rule unknown")

        return potential_payment * np.where(allocation > 0, 1.0, 0.0)

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

    # --------------------------------- methods to compute metrics --------------------------------- #

    def get_metrics(
        self, agent: str, obs_profile: np.ndarray, bid_profile: np.ndarray
    ) -> tuple:
        """compute all metrics relevant for single-item auction
        this includes standard metrics (l2, util_loss) + revenue
        """
        metrics, values = self.get_standard_metrics(agent, obs_profile, bid_profile)
        metrics += ["revenue"]
        values += [self.compute_revenue(obs_profile, bid_profile)]
        metrics += ["efficiency"]
        values += [self.compute_efficiency(obs_profile, bid_profile)]
        return metrics, values

    def compute_revenue(
        self, obs_profile: np.ndarray, bid_profile: np.ndarray
    ) -> float:
        """Compute expected revenue for LLG Model

        Args:
            bid_profile (np.ndarray): bid profile

        Returns:
            float: approximated expected revenue
        """
        allocations = np.array(
            [self.get_allocation(bid_profile, i) for i in range(self.n_bidder)]
        )
        payments = np.array(
            [
                self.get_payment(bid_profile, allocations[i], i)
                for i in range(self.n_bidder)
            ]
        )
        welfare = (allocations * obs_profile).sum(axis=0)
        revenue = (allocations * payments).sum(axis=0)
        return revenue.mean() / welfare.mean()

    def compute_efficiency(
        self, obs_profile: np.ndarray, bid_profile: np.ndarray
    ) -> float:
        """_summary_

        Args:
            obs_profile (np.ndarray): valuation profiles
            bid_profile (np.ndarray): bid profiles

        Returns:
            float: efficiency (percentage of possible total welfare)
        """
        allocations_truthful = np.array(
            [self.get_allocation(obs_profile, i) for i in range(self.n_bidder)]
        )
        allocations_bids = np.array(
            [self.get_allocation(bid_profile, i) for i in range(self.n_bidder)]
        )
        welfare_truthful = (allocations_truthful * obs_profile).sum(axis=0)
        welfare_bids = (allocations_bids * obs_profile).sum(axis=0)
        return welfare_bids.mean() / welfare_truthful.mean()

    # ------------------------------- methods to determin equilibria ------------------------------- #

    def get_bne(self, agent: str, obs: np.ndarray) -> np.ndarray:
        """
        Returns BNE for different payment rulse (assuming observations and actions for local bidders are [0,1])

        Args:
            agent (str): specifies bidder, either local L or global G
            obs (np.ndarray): _description_

        Returns:
            np.ndarray: equilibrium bid for each observation
        """
        bnes = [
            self.bne_nearest_vcg(agent, obs),
            self.bne_nearest_nz(agent, obs),
            self.bne_nearest_nb(agent, obs),
        ]
        for bne in bnes:
            if bne is not None:
                return bne
        return None

    def bne_nearest_vcg(self, agent: str, obs: np.ndarray):
        """BNE for LLG Auction with Nearest-VCG payment rule"""
        if (self.payment_rule == "nearest_vcg") & (self.prior == "uniform"):
            if agent == "L":
                if self.gamma == 1:
                    return 2 / 3 * obs
                else:
                    obs_star = (3 - np.sqrt(9 - (1 - self.gamma) ** 2)) / (
                        1 - self.gamma
                    )
                    return np.maximum(
                        np.zeros(len(obs)), 2 / (2 + self.gamma) * (obs - obs_star)
                    )
            elif agent == "G":
                return obs
            else:
                raise ValueError("BNE only for local (L) or global (G) bidder.")
        else:
            return None

    def bne_nearest_nz(self, agent: str, obs: np.ndarray):
        """BNE for LLG Auction with Nearest-Zero payment rule"""
        if (self.payment_rule == "nearest_zero") & (self.prior == "uniform"):
            if agent == "L":
                return np.maximum(
                    np.zeros(len(obs)),
                    np.log(self.gamma + (1 - self.gamma) * obs) / (1 - self.gamma) + 1,
                )
            elif agent == "G":
                return obs
            else:
                raise ValueError("BNE only for local (L) or global (G) bidder.")
        else:
            return None

    def bne_nearest_nb(self, agent: str, obs: np.ndarray):
        """BNE for LLG Auction with Nearest-Bid payment rule"""
        if (self.payment_rule == "nearest_bid") & (self.prior == "uniform"):
            if agent == "L":
                if self.gamma == 1:
                    return 1 / 2 * obs
                else:
                    return (
                        1
                        / (1 - self.gamma)
                        * (np.log(2) - np.log(2 - (1 - self.gamma) * obs))
                    )
            elif agent == "G":
                return obs
            else:
                raise ValueError("BNE only for local (L) or global (G) bidder.")
        else:
            return None

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
        elif self.param_util["payment_rule"] not in [
            "nearest_vcg",
            "nearest_zero",
            "nearest_bid",
            "first_price",
        ]:
            raise ValueError("payment rule is not available")

        if self.bidder[2] != "G":
            raise ValueError('choose either ["L","L","G"] or  ["L1","L2","G"] ')

        if "corr" not in self.param_prior:
            self.param_prior["corr"] = 0.0
            print(
                "No correlation for LLG Auction defined. We assume no correlation (corr=0)."
            )

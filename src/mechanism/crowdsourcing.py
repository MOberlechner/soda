from typing import Dict, List

import numpy as np
from scipy.special import binom

from .mechanism import Mechanism

# -------------------------------------------------------------------------------------------------------------------- #
#                                                  CROWDSOURCING                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


class Crowdsourcing(Mechanism):
    """Crowdsourcing - All-Pay Auction with several prices. Valuations of all prices sum up to one.

    prices: list,  of prices in decreasing order, number of prices <= number of bidders
    param_util: dict, contains "tiebreaking"-rule and (optional) "util_type" (cost (default) or valuation)

    """

    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        param_prior: Dict[str, str],
        param_util: Dict[str, float],
    ):
        super().__init__(bidder, o_space, a_space, param_prior, param_util)
        self.name = "crowdsourcing"

        self.check_param()
        self.prices = param_util["prices"]
        self.tie_breaking = param_util["tie_breaking"]
        self.payment_rule = param_util["payment_rule"]
        self.type = param_util["type"]

        self.check_own_gradient()

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int):
        """utility for crowdsourcing contest with prices (decreasing, sum to 1)

        Args:
            obs (np.ndarray): _description_
            bids (np.ndarray): _description_
            idx (int): _description_

        Returns:
            _type_: _description_
        """
        self.test_input_utility(obs, bids, idx)
        valuation = self.get_valuation(obs, bids, idx)
        allocation = self.get_allocation(bids, idx)
        payment = self.get_payment(bids, allocation, idx)
        payoff = self.get_payoff(allocation, valuation, payment)

        return payoff

    def get_payoff(
        self, allocation: np.ndarray, valuation: np.ndarray, payment: np.ndarray
    ) -> np.ndarray:
        """compute payoff given allocation, payment and valuation vector

        Args:
            allocation (np.ndarray): allocation matrix for agent
            valuation (np.ndarray): valuation
            payment (np.ndarray): _description_

        Returns:
            np.ndarray: payoff
        """
        if self.type == "cost":
            return self.prices.dot(allocation) - valuation * payment
        elif self.type == "value":
            return self.prices.dot(allocation) * valuation - payment
        else:
            raise ValueError(f"chose type between value and cost")

    def get_allocation(self, bids: np.ndarray, idx: int) -> np.ndarray:
        """compute allocation given action profiles for agent i

        Args:
            bids (np.ndarray): action profiles
            idx (int): index of agent we consider

        Returns:
            np.ndarray: allocation matrix for agent i
        """
        if self.tie_breaking == "random":
            rank_min = (bids[idx] < bids).sum(axis=0)
            rank_max = self.n_bidder - (bids[idx] > bids).sum(axis=0) - 1
            number_bidder_on_same_rank = rank_max - rank_min + 1

            # compute probability matrix (i,j): contains probability of winning price i for each bid j
            probs = np.zeros(bids.shape)
            for i in range(self.n_bidder):
                probs[
                    np.minimum(np.maximum(rank_min, i), rank_max),
                    np.arange(bids.shape[1]),
                ] = 1
            probs = probs / number_bidder_on_same_rank.T

        elif self.tie_breaking == "lose":
            rank_max = self.n_bidder - (bids[idx] > bids).sum(axis=0) - 1

            # compute probability matrix (i,j): contains probability of winning price i for each bid j
            probs = np.zeros(bids.shape)
            probs[rank_max, np.arange(bids.shape[1])] = 1

        elif self.tie_breaking == "win":
            rank_min = (bids[idx] < bids).sum(axis=0)

            # compute probability matrix (i,j): contains probability of winning price i for each bid j
            probs = np.zeros(bids.shape)
            probs[rank_min, np.arange(bids.shape[1])] = 1

        else:
            raise ValueError(f"tie_breaking rule {self.tie_breaking} unknown")

    def get_payment(
        self, bids: np.ndarray, allocation: np.ndarray, idx: int
    ) -> np.ndarray:
        """compute payment for bidder idx

        Args:
            bids (np.ndarray): action profiles
            allocation (np.ndarray): allocation vector for agent i
            idx (int): index of agent we consider

        Returns:
            np.ndarray: payment vector
        """
        if self.payment_rule == "first_price":
            return bids[idx]
        else:
            raise ValueError(f"payment_rule unkown")

    def get_bne(self, agent: str, obs: np.ndarray) -> np.ndarray:
        """returns BNE for some specific settings

        Args:
            agent (str): agents
            obs (np.ndarray): observed types

        Returns:
            np.ndarray: equilibrium bids
        """
        bnes = [
            self.bne_uniform_2_prizes(agent, obs),
        ]
        for bne in bnes:
            if bne is not None:
                return bne
        return None

    def bne_uniform_2_prizes(self, agent: str, obs: np.ndarray) -> np.ndarray:
        """BNE for crowdsourcing contest with 2 prizes and uniform prior (value)"""
        if (
            self.check_bidder_symmetric([0, 1])
            & (self.type == "value")
            & (self.prior == "uniform")
            & ((self.prices > 0).sum() <= 2)
        ):
            bne = self.prices[0] * (
                self.n_bidder - 1
            ) / self.n_bidder * obs**self.n_bidder + self.prices[1] * (
                (self.n_bidder - 2) * obs ** (self.n_bidder - 1)
                - (self.n_bidder - 1) ** 2 / self.n_bidder * obs**self.n_bidder
            )
            return bne
        else:
            return None

    def compute_gradient(self, game, strategies, agent: str):
        """_summary_

        Args:
            game (_type_): _description_
            strategies (_type_): _description_
            agent (str): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        pdf = strategies[agent].x.sum(axis=0)
        cdf = np.insert(pdf, 0, 0.0).cumsum()
        exp_win = np.array(
            [
                self.prices[r]
                * binom(self.n_bidder - 1, r)
                * cdf[:-1] ** (self.n_bidder - r - 1)
                * (1 - cdf[1:]) ** r
                for r in range(sum(self.prices > 0))
            ]
        ).sum(axis=0)

        if self.type == "cost":
            payment = (
                strategies[agent]
                .o_discr.reshape(game.n, 1)
                .dot(strategies[agent].a_discr.reshape(1, game.m))
            )
            return exp_win - payment

        elif self.type == "valuation":
            exp_win_val = (
                strategies[agent]
                .o_discr.reshape(strategies[agent].n, 1)
                .dot(exp_win.reshape(1, game.m))
            )
            payment = np.ones((game.n, 1)).dot(
                strategies[agent].a_discr.reshape(1, game.m)
            )
            return exp_win_val - payment

        else:
            raise ValueError("util_type in param_util unknown")

    def check_param(self):
        """
        Check if input paremter are sufficient to define mechanism
        """
        if "tie_breaking" not in self.param_util:
            raise ValueError("specify tiebreaking in param_util")

        if "type" not in self.param_util:
            raise ValueError("specify param_util - type: cost or value")

        if "prices" not in self.param_util:
            raise ValueError("specify prices in param_util")
        else:
            if not np.isclose(sum(self.param_util["prices"]), 1.0):
                raise ValueError("prices have to add up to one")
            else:
                n_prices = len(self.param_util["prices"])
                if n_prices > self.n_bidder:
                    raise ValueError("More prices than bidders")
                elif n_prices < self.n_bidder:
                    self.param_util["prices"] = self.param_util["prices"] + [0] * (
                        self.n_bidder - n_prices
                    )
                else:
                    pass

        if "payment_rule" not in self.param_util:
            self.param_util["payment_rule"] = "first_price"
        else:
            if self.param_util["payment_rule"] != "first_price":
                raise NotImplementedError(
                    "only first_price payment_rule is implemented"
                )

    def check_own_gradient(self):
        """check if we cna use simplified gradient computation of mechanism"""
        if (
            self.check_bidder_symmetric()
            & ("corr" not in self.param_prior)
            & (self.payment_rule == "first_price")
            & (self.tie_breaking == "lose")
        ):
            self.own_gradient = True

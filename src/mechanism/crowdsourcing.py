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
        self.prices = np.array(
            param_util["prices"]
            if len(param_util["prices"]) == len(bidder)
            else param_util["prices"] + [0] * (len(bidder) - len(param_util["prices"]))
        )
        self.param_util = param_util
        self.type = param_util["type"] if "type" in param_util else "cost"
        self.own_gradient = (len(self.set_bidder) == 1) and (
            self.param_util["tiebreaking"] == "lose"
        )

        # check input
        if self.prices.sum() != 1:
            raise ValueError("Prices have to add up to one")

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int):
        """Crowdsourcing Contest: Given Prices [p_1, p_2, ..., p_N] (in decrasing order with sum p_i = 1), the bidder
        wins the price corresponding to his or her rank. Tie-breaking rule is random


        Parameters
        ----------
        obs : obs corresponds to skill paramater
        bids : effort
        idx : agent

        Returns
        -------

        """
        # test input
        if bids.shape[0] != self.n_bidder:
            raise ValueError("wrong format of bids")
        elif idx >= self.n_bidder:
            raise ValueError("bidder with index " + str(idx) + " not avaible")
        elif "tiebreaking" not in self.param_util:
            raise ValueError("specify tiebreaking rule")
        else:
            pass

        # if True: we want each outcome for every observation, else: each outcome belongs to one observation
        if obs.shape != bids[idx].shape:
            obs = obs.reshape(len(obs), 1)

        # determine allocation (different tie breaking rules)
        if self.param_util["tiebreaking"] == "random":
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

        elif self.param_util["tiebreaking"] == "index":
            # compute probability matrix (i,j): contains probability of winning price i for each bid j (only 0/1)
            probs = np.argsort(-bids, axis=0) == idx

        elif self.param_util["tiebreaking"] == "lose":
            rank_max = self.n_bidder - (bids[idx] > bids).sum(axis=0) - 1

            # compute probability matrix (i,j): contains probability of winning price i for each bid j
            probs = np.zeros(bids.shape)
            probs[rank_max, np.arange(bids.shape[1])] = 1

        elif self.param_util["tiebreaking"] == "win":
            rank_min = (bids[idx] < bids).sum(axis=0)

            # compute probability matrix (i,j): contains probability of winning price i for each bid j
            probs = np.zeros(bids.shape)
            probs[rank_min, np.arange(bids.shape[1])] = 1

        elif self.param_util["tiebreaking"] == "zero":
            rank_min = (bids[idx] < bids).sum(axis=0)
            rank_max = self.n_bidder - (bids[idx] > bids).sum(axis=0) - 1
            rank = np.where(rank_max == rank_min, rank_min, self.n_bidder - 1)

            # compute probability matrix (i,j): contains probability of winning price i for each bid j
            probs = np.zeros(bids.shape)
            for i in range(self.n_bidder):
                probs[rank, np.arange(bids.shape[1])] = 1

        else:
            raise ValueError("tiebreaking rule unknown (random/index/lose/win)")

        # utility: type is marginal cost with linear cost function
        if self.type == "cost":
            return self.prices.dot(probs) - obs * bids[idx]
        elif self.type == "valuation":
            return self.prices.dot(probs) * obs - bids[idx]
        else:
            raise ValueError("util_type in param_util unknown")

    def compute_gradient(self, strategies, game, agent: str):
        """Simplified computation of gradient for i.i.d. bidders and tie-breaking "lose"

        Parameters
        ----------
        strategies :
        game :
        agent :

        Returns
        -------

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

    def get_bne(self, agent: str, obs: np.ndarray):

        if (
            self.o_space[agent] == [0, 1]
            and self.type == "valuation"
            and len(self.set_bidder) == 1
            and (self.prices > 0).sum() <= 2
        ):

            bids_bne = self.prices[0] * (
                self.n_bidder - 1
            ) / self.n_bidder * obs**self.n_bidder + self.prices[1] * (
                (self.n_bidder - 2) * obs ** (self.n_bidder - 1)
                - (self.n_bidder - 1) ** 2 / self.n_bidder * obs**self.n_bidder
            )
            return bids_bne

        else:
            raise NotImplementedError("BNE for this setting not implemented")

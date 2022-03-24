from typing import Dict, List

import numpy as np

from .mechanism import Mechanism

# -------------------------------------------------------------------------------------------------------------------- #
#                                                   CONTEST GAME                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


class Crowdsourcing(Mechanism):
    """ Crowdsourcing - All-Pay Auction with several prices. Valuations of all prices sum up to one.

    """

    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        param_prior: Dict[str, str],
        prices: List[float],
        param_util: Dict[str, float],
    ):
        super().__init__(bidder, o_space, a_space, param_prior)
        self.name = "crowdsourcing"
        self.prices = (
            prices
            if len(prices) == len(bidder)
            else prices + [0] * (len(bidder) - len(prices))
        )
        self.param_util = param_util

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int):
        """ Crowdsourcing Contest: Given Prices [p_1, p_2, ..., p_N] (in decrasing order with sum p_i = 1), the bidder
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
        return np.array(self.prices).dot(probs) - obs * bids[idx]

    def get_bne(self, agent: str, obs: np.ndarray):

        pass

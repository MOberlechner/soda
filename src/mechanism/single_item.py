from typing import Dict, List

import numpy as np

from .mechanism import Mechanism

# -------------------------------------------------------------------------------------------------------------------- #
#                                             SINGLE-ITEM AUCTION                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


class SingleItemAuction(Mechanism):
    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        prior: str,
        payment_rule: str = "first_price",
        risk: float = 1.0,
    ):
        super().__init__(bidder, o_space, a_space, prior)
        self.payment_rule = payment_rule
        self.risk = risk

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int):
        """ Payoff function for first price sealed bid auctons

        Parameters
        ----------
        obs : observation/valuation of bidder (idx)
        bids : array with bid profiles
        idx : index of bidder to consider

        Returns
        -------

        """

        # test input
        if bids.shape[0] != self.n_bidder:
            raise ValueError("wrong format of bids")
        elif idx >= self.n_bidder:
            raise ValueError("bidder with index " + str(idx) + " not avaible")

        # if True: we want each outcome for every observation,  each outcome belongs to one observation
        if obs.shape != bids[idx].shape:
            obs = obs.reshape(len(obs), 1)

        # determine winner (and number of winner for tie-breaking)
        win = np.where(bids[idx] > np.delete(bids, idx, 0).max(axis=0), 1, 0)
        num_winner = (bids.max(axis=0) == bids).sum(axis=0)

        if self.payment_rule == "first_price":
            payoff = obs - bids[idx]
            return 1 / num_winner * win * np.sign(payoff) * np.abs(payoff) ** self.risk

        elif self.payment_rule == "second_price":
            payoff = obs - np.delete(bids, idx, 0).max(axis=0)
            return 1 / num_winner * win * np.sign(payoff) * np.abs(payoff) ** self.risk
        else:
            raise ValueError("payment rule " + self.payment_rule + " not available")

    def bids_bne(self, obs: np.ndarray):
        if self.prior == "uniform":
            if (self.payment_rule == "first_price") & np.all(
                [self.o_space[i] == [0, 1] for i in self.set_bidder]
            ):
                return (self.n_bidder - 1) / (self.n_bidder - 1 + self.risk) * obs
            elif self.payment_rule == "second_price":
                return obs

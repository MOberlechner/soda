from typing import Dict, List

import numpy as np

from .mechanism import Mechanism


class SingleItemAuction(Mechanism):
    def __init__(
        self,
        bidder: List[str],
        v_space: Dict[List],
        a_space: Dict[List],
        prior: str,
        payment_rule: str = "first_price",
        risk: float = 1.0,
    ):
        super().__init__(bidder, v_space, a_space, prior)
        self.payment_rule = payment_rule
        self.risk = risk

    def utility(self, val: np.ndarray, bids: np.ndarray, idx: int):
        """

        Parameters
        ----------
        val : valuation of bidder (idx)
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

        # determine winner (and number of winner for tie-breaking)
        win = np.where(bids[idx] > np.delete(bids, idx, 0).max(axis=0), 1, 0)
        num_winner = (bids.max(axis=0) == bids).sum(axis=0)

        if self.payment_rule == "first_price":
            payoff = val - bids[idx]
            return 1 / num_winner * win * np.sign(payoff) * np.abs(payoff) ** self.risk

        elif self.payment_rule == "second_price":
            payoff = val - np.delete(bids, idx, 0).max(axis=0)
            return 1 / num_winner * win * np.sign(payoff) * np.abs(payoff) ** self.risk
        else:
            raise ValueError("payment rule " + self.payment_rule + " not available")

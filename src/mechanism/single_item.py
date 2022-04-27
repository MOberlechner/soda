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
        param_prior: Dict[str, str],
        param_util: Dict,
    ):
        super().__init__(bidder, o_space, a_space, param_prior)
        self.name = "single_item"
        self.param_util = param_util
        self.payment_rule = param_util["payment_rule"]
        self.risk = param_util["risk"] if "risk" in param_util else 1.0

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
        elif "tie_breaking" not in self.param_util:
            raise ValueError("specify tiebreaking rule")

        # if True: we want each outcome for every observation,  each outcome belongs to one observation
        if obs.shape != bids[idx].shape:
            obs = obs.reshape(len(obs), 1)

        # paramater utility
        if "tie_breaking" not in self.param_util:
            self.param_util["tie_breaking"] = "random"

        # determine allocation
        if self.param_util["tie_breaking"] == "random":
            win = np.where(bids[idx] >= np.delete(bids, idx, 0).max(axis=0), 1, 0)
            num_winner = (bids.max(axis=0) == bids).sum(axis=0)

        elif self.param_util["tie_breaking"] == "lose":
            win = np.where(bids[idx] > np.delete(bids, idx, 0).max(axis=0), 1, 0)
            num_winner = np.ones(win.shape)

        else:
            raise ValueError(
                'Tie-breaking rule "' + self.param_util["tie_breaking"] + '" unknown'
            )

        # determine payoff
        if self.payment_rule == "first_price":
            payoff = obs - bids[idx]
            return 1 / num_winner * win * np.sign(payoff) * np.abs(payoff) ** self.risk

        elif self.payment_rule == "second_price":
            payoff = obs - np.delete(bids, idx, 0).max(axis=0)
            return 1 / num_winner * win * np.sign(payoff) * np.abs(payoff) ** self.risk
        else:
            raise ValueError("payment rule " + self.payment_rule + " not available")

    def get_bne(self, obs: np.ndarray):
        if self.prior == "uniform":
            if (self.payment_rule == "first_price") & np.all(
                [self.o_space[i] == [0, 1] for i in self.set_bidder]
            ):
                return (self.n_bidder - 1) / (self.n_bidder - 1 + self.risk) * obs
            elif self.payment_rule == "second_price":
                return obs

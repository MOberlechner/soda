from typing import Dict, List

import numpy as np

from .mechanism import Mechanism


class DoubleAuction(Mechanism):
    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[List],
        a_space: Dict[List],
        prior: str,
        payment_rule: str = "average",
        risk: float = 1.0,
    ):
        # TODO: Fix Implementation for Double Auction
        super().__init__(bidder, o_space, a_space, param_prior, param_util)
        self.name = "double_auction"
        self.payment_rule = payment_rule
        self.risk = risk

    def utility(self, val: np.ndarray, bids: np.ndarray, idx: int):
        """
        Returns the utility of bidder (idx) in a double auction

        Parameters
        ----------
        val : np.ndarray, valuations
        bids : np.ndarray, bid profiles
        idx : int, position of respective bidder within the lust of bidders

        Returns
        -------
        np.ndarray, shape depends on valuations and bids
        """
        # TODO: tie breaking rule is not yet implemented

        # deterimine parameter
        n_bidder, n_profiles = bids.shape
        n_seller = self.bidder.count("S")
        n_buyer = self.bidder.count("B")

        # test input
        if bids.shape[0] != self.n_bidder:
            raise ValueError("wrong format of bids")
        if n_seller + n_buyer != n_bidder:
            raise ValueError("bidder must only contain buyers (B) or sellers (S)")
        if self.bidder != ["S"] * n_seller + ["B"] * n_buyer:
            raise ValueError(
                "bidder must by given by a list of the form [S,...,S,B,...,B]"
            )

        # determine where trade takes place
        seller = np.array(self.bidder) == "S"
        buyer = np.array(self.bidder) == "B"

        # determine number of traded items
        number_trades = (
            -np.sort(-bids[buyer], axis=0) >= np.sort(bids[seller], axis=0)
        ).sum(axis=0)
        number_trades = np.minimum(number_trades, min(n_bidder, n_seller))

        # does player idx participate in the trade
        idx_within_group = idx - n_seller if self.bidder[idx] == "B" else idx
        trade_bool = (
            np.argsort(bids[np.array(self.bidder) == self.bidder[idx]], axis=0)[
                idx_within_group
            ]
            < number_trades
        )

        if self.payment_rule == "average":
            payment = 0.5 * (
                -np.sort(-bids[buyer], axis=0)[number_trades - 1, np.arange(n_profiles)]
                + np.sort(bids[seller], axis=0)[
                    number_trades - 1, np.arange(n_profiles)
                ]
            )

        elif self.payment_rule == "vcg":
            if self.bidder == ["S", "B"]:
                payment = bids[seller] if self.bidder[idx] == "B" else bids[buyer]

            else:
                payment = None
                print("VCG not yet implemented for more than one seller and one buyer")
        else:
            raise ValueError("payment rule " + self.payment_rule + " not available")

        # if True: we want each outcome for every valuation,  each outcome belongs to one valuation
        if val.shape != bids[idx].shape:
            val = val.reshape(len(val), 1)

        return (
            trade_bool
            * (1 if self.bidder[idx] == "B" else -1)
            * np.sign(val - payment)
            * np.abs(val - payment) ** self.risk
        )

    def get_bne(self, agent: str, obs: np.ndarray):
        # check if
        if (
            (self.prior == "uniform")
            & (self.payment_rule == "average")
            & (self.n_bidder == 2)
        ):
            c = 2 ** (1 / self.risk) - 1 / 2
            if agent == "S":
                return (c - 1 / 2) / (2 * c**2 - 1 / 2) + (1 - 1 / (2 * c)) * obs
            else:
                return (1 - 1 / (2 * c)) / (4 * c**2 - 1) + (1 - 1 / (2 * c)) * obs

        elif self.payment_rule == "vcg":
            return obs
        else:
            raise NotImplementedError("No BNE for non-uniform prior implemented.")

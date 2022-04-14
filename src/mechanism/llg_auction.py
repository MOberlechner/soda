from typing import Dict, List

import numpy as np

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
        prior: str,
        param_util: Dict,
    ):
        super().__init__(bidder, o_space, a_space, prior)
        self.name = "llg_auction"
        self.param_util = param_util
        self.payment_rule = param_util["payment_rule"]
        self.tie_breaking = param_util["tie_breaking"]

        # check input
        if self.bidder[2] != "G":
            raise ValueError('choose either ["L","L","G"] or  ["L1","L2","G"] ')

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int):
        """ Payoff function for first price sealed bid auctons

        Parameters
        ----------
        obs : observation/valuation of bidder (idx)
        bids : array with bid profiles
        idx : index of bidder to consider

        Returns
        -------
        utility for the LLG-Auction under different bidder-optimal core-selecting rules
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

        # determine payoff - global bidder (idx = 2)
        if idx == 2:
            win = bids[2] >= bids[:2].sum(axis=0)
            tie = bids[2] == bids[:2].sum(axis=0) if self.tie_breaking else 0 * win

            return win * (1 - 0.5 * tie) * (obs - bids[:2].sum(axis=0))

        # TODO: Fehler in Implementierung, Utilities überprüfen
        # determine payoff - local bidder (idx = 1,2)
        else:
            if self.payment_rule == "NZ":
                # Nearest-Zero (Proxy Rule)
                win_a = bids[2] <= 2 * bids[:2].min(axis=0)
                win_b = (2 * bids[:2].min(axis=0) < bids[2]) & (
                    bids[2] <= bids[:2].sum(axis=0)
                )
                tie = (
                    bids[2] <= 2 * bids[:2].sum(axis=0)
                    if self.tie_breaking
                    else 0 * win_a
                )

                return (1 - 0.5 * tie) * (
                    win_a * (obs - 0.5 * bids[2])
                    + win_b
                    * (
                        obs
                        - np.where(
                            bids[idx] == bids[:2].min(axis=0), bids[idx], bids[2]
                        )
                        - bids[idx]
                    )
                )

            elif self.payment_rule == "NVCG":
                # Nearest-VCG Rule
                win = bids[2] <= bids[:2].sum(axis=0)
                tie = bids[2] == bids[:2].sum(axis=0) if self.tie_breaking else 0 * win
                # compute VCG payments
                p_vcg = {
                    0: -bids[1] + np.maximum(bids[1], bids[2]),
                    1: -bids[0] + np.maximum(bids[0], bids[2]),
                }

                delta = 0.5 * (bids[2] - p_vcg[0] - p_vcg[1])
                return win * (1 - 0.5 * tie) * (obs - p_vcg[idx] - delta)

            elif self.payment_rule == "NB":
                # Nearest-Bid Rule
                win_a = bids[2] <= bids[:2].max(axis=0) - bids[:2].min(axis=0)
                win_b = (bids[2] > bids[:2].max(axis=0) - bids[:2].min(axis=0)) & (
                    bids[2] <= bids[:2].sum(axis=0)
                )
                tie = (
                    bids[2] == bids[:2].sum(axis=0) if self.tie_breaking else 0 * win_a
                )

                delta = 0.5 * (bids[:2].sum(axis=0) - bids[2])
                return (1 - 0.5 * tie) * (
                    win_a
                    * (obs - np.where(bids[idx] == bids[:2].max(axis=0), bids[2], 0))
                    + win_b * (obs - bids[idx] + delta)
                )

    def get_bne(self, obs: np.ndarray):
        if self.prior == "uniform":
            if (self.payment_rule == "first_price") & np.all(
                [self.o_space[i] == [0, 1] for i in self.set_bidder]
            ):
                return (self.n_bidder - 1) / (self.n_bidder - 1 + self.risk) * obs
            elif self.payment_rule == "second_price":
                return obs

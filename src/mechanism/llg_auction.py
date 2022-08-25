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
        param_prior: Dict,
        param_util: Dict,
    ):

        super().__init__(bidder, o_space, a_space, param_prior, param_util)
        self.name = "llg_auction"
        self.param_util = param_util
        self.payment_rule = param_util["payment_rule"]
        self.tie_breaking = (
            param_util["tie_breaking"] if "tie_breaking" in param_util else None
        )
        if "corr" in param_prior:
            self.gamma = param_prior["corr"]
        else:
            self.gamma = 0.0
            print("No correlation for LLG Auction defined (gamma=0).")

        # check input
        if self.bidder[2] != "G":
            raise ValueError('choose either ["L","L","G"] or  ["L1","L2","G"] ')

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

        # test input
        if bids.shape[0] != self.n_bidder:
            raise ValueError("wrong format of bids")
        elif idx >= self.n_bidder:
            raise ValueError("bidder with index " + str(idx) + " not avaible")
        elif self.tie_breaking is None:
            raise ValueError("specify tiebreaking rule")
        elif self.tie_breaking not in ["random", "local", "lose"]:
            raise ValueError("specified tiebreaking rule unkown")

        # if True: we want each outcome for every observation,  each outcome belongs to one observation
        if obs.shape != bids[idx].shape:
            obs = obs.reshape(len(obs), 1)

        # tie-breaking rule
        tie_local = (
            bids[2]
            == bids[:2].sum(axis=0)
            * {"random": 0.5, "local": 0.0, "lose": 1.0}[self.tie_breaking]
        )
        tie_global = (
            bids[2]
            == bids[:2].sum(axis=0)
            * {"random": 0.5, "local": 1.0, "lose": 1.0}[self.tie_breaking]
        )

        # determine payoff - global bidder (idx = 2)
        if idx == 2:
            win = bids[2] >= bids[:2].sum(axis=0)
            return win * (1 - tie_global) * (obs - bids[:2].sum(axis=0))

        # determine payoff - local bidder (idx = 1,2)
        else:
            # Nearest-Zero (Proxy Rule)
            if self.payment_rule == "NZ":

                win_a = bids[2] <= 2 * bids[:2].min(axis=0)
                win_b = (2 * bids[:2].min(axis=0) < bids[2]) & (
                    bids[2] <= bids[:2].sum(axis=0)
                )
                return (1 - tie_local) * (
                    win_a * (obs - 0.5 * bids[2])
                    + win_b
                    * (
                        obs
                        - np.where(
                            bids[idx] == bids[:2].min(axis=0),
                            bids[idx],
                            bids[2] - bids[:2].min(axis=0),
                        )
                    )
                )

            # Nearest-VCG Rule
            elif self.payment_rule == "NVCG":

                win = bids[2] <= bids[:2].sum(axis=0)
                payments_vcg = {
                    0: -bids[1] + np.maximum(bids[1], bids[2]),
                    1: -bids[0] + np.maximum(bids[0], bids[2]),
                }
                delta = 0.5 * (bids[2] - payments_vcg[0] - payments_vcg[1])
                return win * (1 - tie_local) * (obs - payments_vcg[idx] - delta)

            # Nearest-Bid Rule
            elif self.payment_rule == "NB":

                win_a = bids[2] <= bids[:2].max(axis=0) - bids[:2].min(axis=0)
                win_b = (bids[2] > bids[:2].max(axis=0) - bids[:2].min(axis=0)) & (
                    bids[2] <= bids[:2].sum(axis=0)
                )

                delta = 0.5 * (bids[:2].sum(axis=0) - bids[2])
                return (1 - tie_local) * (
                    win_a
                    * (obs - np.where(bids[idx] == bids[:2].max(axis=0), bids[2], 0))
                    + win_b * (obs - bids[idx] + delta)
                )

    def get_bne(self, agent: str, obs: np.ndarray):
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

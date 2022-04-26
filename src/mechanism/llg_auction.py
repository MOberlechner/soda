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
        param_prior: Dict[str, str],
        param_util: Dict,
    ):
        """

        Parameters
        ----------
        bidder : list, contains strings, use either ["L", "L", "G"] or ["L1", "L2", "G2]
        o_space :
        a_space :
        param_prior :
        param_util :    tie_breaking (bool), True = random, False = local always win
                        payment_rule (str), chose core-selecting payment rule: NZ, NVCG, NB
        """
        super().__init__(bidder, o_space, a_space, param_prior)
        self.name = "llg_auction"
        self.param_util = param_util
        self.payment_rule = param_util["payment_rule"]
        self.tie_breaking = (
            param_util["tie_breaking"] if "tie_breaking" in param_util else None
        )
        self.gamma = param_prior["gamma"] if "gamma" in param_prior else 0.0

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
            * {"random": 1.0, "local": 0.0, "lose": 2.0}[self.tie_breaking]
        )
        tie_global = (
            bids[2]
            == bids[:2].sum(axis=0)
            * {"random": 1.0, "local": 2.0, "lose": 2.0}[self.tie_breaking]
        )

        # determine payoff - global bidder (idx = 2)
        if idx == 2:
            win = bids[2] >= bids[:2].sum(axis=0)
            return win * (1 - 0.5 * tie_global) * (obs - bids[:2].sum(axis=0))

        # determine payoff - local bidder (idx = 1,2)
        else:
            if self.payment_rule == "NZ":
                # Nearest-Zero (Proxy Rule)
                win_a = bids[2] <= 2 * bids[:2].min(axis=0)
                win_b = (2 * bids[:2].min(axis=0) < bids[2]) & (
                    bids[2] <= bids[:2].sum(axis=0)
                )
                return (1 - 0.5 * tie_local) * (
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
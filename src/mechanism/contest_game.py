from typing import Dict, List

import numpy as np

from .mechanism import Mechanism

# -------------------------------------------------------------------------------------------------------------------- #
#                                                   CONTEST GAME                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


class ContestGame(Mechanism):
    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        param_prior: Dict[str, str],
        csf: str = "allpay",
        param_csf: float = 0.0,
    ):
        super().__init__(bidder, o_space, a_space, param_prior)
        self.csf = csf
        self.param_csf = param_csf

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int):
        """ We consider different contest success functions
        - allpay: the winner (random tie breaking rule) gets the item and everyone pays own bid

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

        # if True: we want each outcome for every observation,  each outcome belongs to one observation
        if obs.shape != bids[idx].shape:
            obs = obs.reshape(len(obs), 1)

        if self.csf == "allpay":
            # determine allocation
            win = np.where(bids[idx] > np.delete(bids, idx, 0).max(axis=0), 1, 0)
            num_winner = (bids.max(axis=0) == bids).sum(axis=0)

            if self.n_bidder == 2:
                # using the homotopy between first-price (param=0) and second-price (param=1)
                # from [Aumann&Leininger, 1996] for two bidders
                return (
                    1 / num_winner * win * obs
                    - (1 - self.param_csf) * bids[idx]
                    - self.param_csf * bids[1 - idx]
                )
            else:
                return 1 / num_winner * win * obs - bids[idx]

        elif self.csf == "ratio_form":
            # determine probabilities
            r = self.param_ratio_form
            prob = bids[idx] ** r / (bids ** r).sum(axis=0)
            return prob * obs - bids[idx]

        elif self.csf == "logistic_ratio_form":
            mu = self.param_ratio_form
            prob = np.exp(mu * bids[idx]) / np.exp(mu * bids[idx]).sum(axis=0)
            return prob * obs - bids[idx]

        else:
            raise ValueError(
                "Content success function (csf) " + self.csf + " not defined"
            )

    def get_bne(self, agent: str, obs: np.ndarray):
        if (self.n_bidder == 2) & (len(self.set_bidder) == 1):
            # two player symmetric case
            if self.csf == "allpay":
                if (self.prior == "uniform") & (self.o_space[self.bidder[0]] == [0, 1]):
                    return 1 / 2 * obs ** 2
                elif (self.prior == "powerlaw") & (
                    self.o_space[self.bidder[0]] == [0, 1]
                ):
                    power = self.param_prior["power"]
                    return power / (power + 1) * obs ** (power + 1)
            else:
                return None
        else:
            return None

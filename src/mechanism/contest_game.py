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
        csf: str = "ratio_form",
        param_csf: float = 0.0,
    ):
        super().__init__(bidder, o_space, a_space, param_prior, param_util)
        self.name = "contest_game"
        self.csf = csf
        self.param_csf = param_csf

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int):
        """We consider different contest success functions
        Perfectly Discriminiating Contests
        - allpay: the winner (random tie breaking rule) gets the item and everyone pays own bid


        Parameters
        ----------
        obs : obs corresponds to marginal cost
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

        if self.csf == "ratio_form_cost":
            # determine probabilities
            r = self.param_csf
            prob = (bids[idx] ** r + np.where(bids.sum(axis=0) == 0, 1, 0)) / (
                (bids**r).sum(axis=0)
                + np.where(bids.sum(axis=0) == 0, self.n_bidder, 0)
            )
            return prob - obs * bids[idx]

        elif self.csf == "ratio_form_valuation":
            # determine probabilities
            r = self.param_csf
            prob = (bids[idx] ** r + np.where(bids.sum(axis=0) == 0, 1, 0)) / (
                (bids**r).sum(axis=0)
                + np.where(bids.sum(axis=0) == 0, self.n_bidder, 0)
            )
            return obs * prob - bids[idx]

        elif self.csf == "difference_form_cost":
            mu = self.param_csf
            prob = np.exp(mu * bids[idx]) / np.exp(mu * bids).sum(axis=0)
            return prob - obs * bids[idx]

        elif self.csf == "difference_form_valuation":
            mu = self.param_csf
            prob = np.exp(mu * bids[idx]) / np.exp(mu * bids).sum(axis=0)
            return obs * prob - bids[idx]

        else:
            raise ValueError(
                "Content success function (csf) " + self.csf + " not defined"
            )

    def get_bne(self, agent: str, obs: np.ndarray):
        pass

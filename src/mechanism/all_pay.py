from typing import Dict, List

import numpy as np

from .mechanism import Mechanism

# -------------------------------------------------------------------------------------------------------------------- #
#                                                   CONTEST GAME                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


class AllPay(Mechanism):
    """ All-pay auction, with parameter param_allpay that determines the cost if bidder wins
    param_allpay = 0 : first-price
    param_allpay = 1 : second-price (war of attrition)
    param_allpay in (0,1) : convex combination of both

    """

    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        param_prior: Dict[str],
        param_allpay: float = 0.0,
    ):
        super().__init__(bidder, o_space, a_space, param_prior)
        self.param_allpay = param_allpay

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int):
        """ We consider different contest success functions
        Perfectly Discriminiating Contests
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

        # determine allocation
        win = np.where(bids[idx] > np.delete(bids, idx, 0).max(axis=0), 1, 0)
        num_winner = (bids.max(axis=0) == bids).sum(axis=0)

        if self.param_allpay == 0.0 or self.n_bidder > 2:
            return 1 / num_winner * win * obs - bids[idx]

        else:
            # using the homotopy between first-price (param=0) and second-price (param=1)
            # from [Aumann&Leininger, 1996] for two bidders
            return (
                1
                / num_winner
                * win
                * (
                    obs
                    - (1 - self.param_allpay) * bids[idx]
                    - self.param_allpay * bids[1 - idx]
                )
                - (1 - win) * bids[idx]
            )

    def get_bne(self, agent: str, obs: np.ndarray):

        if (self.n_bidder == 2) & (len(self.set_bidder) == 1):

            # symmatric 2 player all pay auction with uniform prior
            if (self.prior == "uniform") & (self.o_space[self.bidder[0]] == [0, 1]):
                if self.param_allpay == 0:
                    return 1 / 2 * obs ** 2
                elif (self.param_allpay > 0) & (self.param_allpay < 1):
                    return (
                        -obs / self.param_allpay
                        - 1
                        / self.param_allpay ** 2
                        * np.log(1 - self.param_allpay * obs)
                    )
                else:
                    return None

            # symmatric 2 player all pay auction with powerlaw prior
            elif (self.prior == "powerlaw") & (self.o_space[self.bidder[0]] == [0, 1]):
                power = self.param_prior["power"]
                return power / (power + 1) * obs ** (power + 1)
        else:
            return None

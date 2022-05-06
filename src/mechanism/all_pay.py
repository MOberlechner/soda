from typing import Dict, List

import numpy as np

from .mechanism import Mechanism

# -------------------------------------------------------------------------------------------------------------------- #
#                                                  ALL-PAY AUCTION                                                     #
# -------------------------------------------------------------------------------------------------------------------- #


class AllPay(Mechanism):
    """All-pay auction, with parameter param_allpay that determines the cost if bidder wins
    param_allpay = 0 : first-price
    param_allpay = 1 : second-price (war of attrition)
    param_allpay in (0,1) : convex combination of both

    """

    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        param_prior: Dict[str, str],
        util_setting: str,
        param_util: Dict[str, float],
    ):
        super().__init__(bidder, o_space, a_space, param_prior, param_util)
        self.name = "all_pay"
        self.util_setting = util_setting

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int):
        """
        - allpay: the winner (random tie breaking rule) gets the item and pays either his own bid (first price),
         the second highest bid (second price) or a convex combination of those two. The loster pay their own bid.


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
        elif "type" not in self.param_util:
            raise ValueError("specify param_util - type: cost or valuation")
        elif "tiebreaking" not in self.param_util:
            raise ValueError("specify tiebreaking rule")
        else:
            pass

        # if True: we want each outcome for every observation,  each outcome belongs to one observation
        if obs.shape != bids[idx].shape:
            obs = obs.reshape(len(obs), 1)

        # determine allocation (random tie breaking rule)
        if self.param_util["tiebreaking"] == "random":
            win = np.where(bids[idx] >= bids.max(axis=0), 1, 0)
            num_winner = (bids.max(axis=0) == bids).sum(axis=0)

        elif self.param_util["tiebreaking"] == "index":
            if idx == 0:
                win = np.where(bids[idx] >= np.delete(bids, idx, 0).max(axis=0), 1, 0)
                num_winner = np.ones(win.shape)
            else:
                win = np.where(bids[idx] > np.delete(bids, idx, 0).max(axis=0), 1, 0)
                num_winner = np.ones(win.shape)

        elif self.param_util["tiebreaking"] == "lose":
            win = bids[idx] > np.delete(bids, idx, 0).max(axis=0)
            num_winner = np.ones(win.shape)
        else:
            raise ValueError("tiebreaking rule unknown (random/index/lose)")

        if self.util_setting == "first_price":
            if self.param_util["type"] == "valuation":
                return obs * win * 1 / num_winner - bids[idx]
            elif self.param_util["type"] == "cost":
                return win * 1 / num_winner - obs * bids[idx]
            else:
                raise ValueError("type in param_util unknown")

        elif self.util_setting == "generalized":
            lamb = self.param_util["price_rule"]
            if self.n_bidder > 2:
                raise ValueError(
                    "For more than two bidders only first-price rule available"
                )

            else:
                # using the homotopy between first-price (param=0) and second-price (param=1)
                # from [Aumann&Leininger, 1996] for two bidders
                return (
                    win
                    * (obs / num_winner - (1 - lamb) * bids[idx] - lamb * bids[1 - idx])
                    - (1 - win) * bids[idx]
                )

        elif self.util_setting == "loss_aversion":

            lamb = self.param_util["lambda"]
            eta = self.param_util["eta"]

            return (
                # bidder is winning
                1 / num_winner * win * (obs - bids[idx] + eta * obs)
                # bidder is losing bc of tiebreaking
                + (num_winner - 1) / num_winner * win * (-bids[idx] - eta * lamb * obs)
                # bidder is losing
                + (1 - win) * (-bids[idx] - eta * lamb * obs)
            )

        elif self.util_setting == "loss_aversion_simple":

            lamb = self.param_util["lambda"]

            return (
                # bidder is winning
                win * 1 / num_winner * (obs - bids[idx])
                # bidder is losing (strictly losing + losing bc of tie breaking
                + ((1 - win) + win * (num_winner - 1) / num_winner)
                * (-lamb * bids[idx])
            )

        else:
            raise ValueError('util_setting "' + self.util_setting + '" unknown')

    def get_bne(self, agent: str, obs: np.ndarray):

        if (self.n_bidder == 2) & (len(self.set_bidder) == 1):
            pass
            """
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
            """

        # BNE for uniform prior over [0,1], 2 bidders and the simple loss aversion model
        if (
            (self.util_setting == "loss_aversion_simple")
            & (self.n_bidder == 2)
            & (self.prior == "uniform")
            & (self.o_space[self.bidder[0]] == [0, 1])
        ):

            lamb = self.param_util["lambda"]
            return 1 / 2 * 1 / (lamb - (lamb - 1) * obs) * obs**2

        else:
            return None

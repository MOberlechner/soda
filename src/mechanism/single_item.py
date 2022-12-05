from typing import Dict, List

import numpy as np
from scipy.special import binom

from .mechanism import Mechanism

# -------------------------------------------------------------------------------------------------------------------- #
#                                             SINGLE-ITEM AUCTION                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


class SingleItemAuction(Mechanism):
    """Single-Item Auction

    Parameter Mechanism
        bidder, o_space, a_space - standard input for all mechanism (see class Mechanism)


    Parameter Prior (param_prior)
        distribution    str: "affiliated_values" and "common_value" are available


    Parameter Utility (param_util)
        tiebreaking     str: specifies tiebreaking rule: "random" (default), "lose"
        payment_rule    str: choose betweem "first_price" and "second_price"
        utility_type    str: QL (quasi-linear), ROI (return of investment), ROS (return of something)

    """

    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        param_prior: Dict[str, str],
        param_util: Dict,
    ):
        super().__init__(bidder, o_space, a_space, param_prior, param_util)
        self.name = "single_item"

        # check input
        if "tie_breaking" not in self.param_util:
            raise ValueError("specify tiebreaking rule")
        elif "payment_rule" not in self.param_util:
            raise ValueError("specify payment rule")
        elif "utility_type" not in self.param_util:
            self.param_prior["utility_type"] = "QL"
            print("utility type not specified, quasi-linear (QL) chosen by default.")

        self.payment_rule = param_util["payment_rule"]
        self.tie_breaking = param_util["tie_breaking"]
        self.utility_type = param_util["utility_type"]

        # prior
        if self.prior in ["affiliated_values", "common_value"]:
            raise NotImplementedError

        # use own gradient
        if (len(self.set_bidder) == 1) and (self.payment_rule == "first_price") & (
            self.prior not in ["affiliated_values", "common_value"]
        ) & ("corr" not in self.param_prior):
            self.own_gradient = True
            print("own gradient")

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int) -> None:
        """
        Payoff function for first price sealed bid auctons

        Parameters
        ----------
        obs : observation/valuation of bidder (idx)
        bids : array with bid profiles
        idx : index of bidder to consider

        Returns
        -------
        np.ndarray : payoff vector for agent idx

        """

        # test input
        if bids.shape[0] != self.n_bidder:
            raise ValueError("wrong format of bids")
        elif idx >= self.n_bidder:
            raise ValueError("bidder with index " + str(idx) + " not avaible")

        # if True: we want each outcome for every observation, else: each outcome belongs to one observation
        if self.values == "private":
            if obs.shape != bids[idx].shape:
                obs = obs.reshape(len(obs), 1)
        else:
            raise ValueError('value model "{}" unknown'.format(self.values))

        # determine allocation
        if self.tie_breaking == "random":
            win = np.where(bids[idx] >= np.delete(bids, idx, 0).max(axis=0), 1, 0)
            num_winner = (bids.max(axis=0) == bids).sum(axis=0)

        elif self.tie_breaking == "lose":
            win = np.where(bids[idx] > np.delete(bids, idx, 0).max(axis=0), 1, 0)
            num_winner = np.ones(win.shape)
        else:
            raise ValueError(
                'Tie-breaking rule "' + self.param_util["tie_breaking"] + '" unknown'
            )

        # determine price
        if self.payment_rule == "first_price":
            price = bids[idx]
        elif self.payment_rule == "second_price":
            price = np.delete(bids, idx, 0).max(axis=0)
        else:
            raise ValueError("payment rule " + self.payment_rule + " not available")

        # utility type
        if self.utility_type == "QL":
            payoff = obs - price
        elif self.utility_type == "ROI":
            payoff = np.divide(
                obs - price, price, out=np.zeros_like((obs - price)), where=price != 0
            )
        elif self.utility_type == "ROS":
            payoff = np.divide(
                obs,
                price,
                out=np.zeros_like((obs / np.ones(price.shape))),
                where=price != 0,
            )
        elif self.utility_type == "ROSB":
            payoff = np.divide(
                obs,
                price,
                out=np.zeros_like((obs / np.ones(price.shape))),
                where=price != 0,
            ) + np.log(self.param_util["budget"] - price)
        else:
            raise ValueError("utility type " + self.utility_type + " not available")

        return 1 / num_winner * win * payoff

    def get_bne(self, agent: str, obs: np.ndarray):
        """
        Returns BNE for some predefined settings

        Parameters
        ----------
        agent : specficies bidder (important in asymmetric settings)
        obs :  observation/valuation of agent

        Returns
        -------
        np.ndarray : bids to corresponding observation

        """
        if self.utility_type == "QL":
            if self.prior == "uniform":
                if (self.payment_rule == "first_price") & np.all(
                    [self.o_space[i] == [0, 1] for i in self.set_bidder]
                ):
                    return (
                        (self.n_bidder - 1) / (self.n_bidder - 1 + self.risk[0]) * obs
                    )
                elif self.payment_rule == "second_price":
                    return obs

            elif self.prior == "affiliated_values":
                return 2 / 3 * obs

            elif self.prior == "common_value":
                return 2 * obs / (2 + obs)

        elif self.utility_type == "ROI":
            if (
                (self.prior == "uniform")
                & (self.payment_rule == "first_price")
                & (len(self.set_bidder == 1))
                & (self.a_space[self.set_bidder[0]][0] > 0)
            ):
                reserve_price = self.a_space[self.set_bidder[0]][0]
                raise NotImplementedError(
                    "Implement BNE for ROI with uniform prior and reserve price"
                )

    def compute_gradient(self, game, strategies, agent: str):
        """Simplified computation of gradient for i.i.d. bidders and tie-breaking "lose"

        Parameters
        ----------
        strategies :
        game :
        agent :

        Returns
        -------

        """
        pdf = strategies[agent].x.sum(axis=0)
        cdf = np.insert(pdf, 0, 0.0).cumsum()[:-1]
        exp_win = cdf ** (self.n_bidder - 1)

        if self.tie_breaking == "lose":
            pass
        elif self.tie_breaking == "random":
            exp_win += sum(
                [
                    binom(self.n_bidder - 1, i)
                    * cdf ** (self.n_bidder - i - 1)
                    / (i + 1)
                    * pdf**i
                    for i in range(1, self.n_bidder)
                ]
            )
        else:
            raise ValueError('Tie-breaking rule "{}" unknown'.format(self.tie_breaking))

        # utility type
        obs_grid = (
            np.ones((strategies[agent].m, strategies[agent].n))
            * strategies[agent].o_discr
        ).T
        bid_grid = (
            np.ones((strategies[agent].n, strategies[agent].m))
            * strategies[agent].a_discr
        )

        if self.utility_type == "QL":
            payoff = obs_grid - bid_grid

        elif self.utility_type == "ROI":
            payoff = np.divide(
                obs_grid - bid_grid,
                bid_grid,
                out=np.zeros_like(obs_grid),
                where=bid_grid != 0,
            )
        elif self.utility_type == "ROS":
            payoff = np.divide(
                obs_grid,
                bid_grid,
                out=np.zeros_like(obs_grid),
                where=bid_grid != 0,
            )
        elif self.utility_type == "ROSB":
            payoff = np.divide(
                obs_grid + np.log(self.param_util["budget"] - bid_grid),
                bid_grid,
                out=np.zeros_like(obs_grid),
                where=bid_grid != 0,
            )
        else:
            raise ValueError(
                "utility type " + self.utility_type + " not available in own gradient"
            )

        return exp_win * payoff

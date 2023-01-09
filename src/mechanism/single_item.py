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
        utility_type    str:    QL (quasi-linear (corresponds to Auction, Default),
                                ROI (return of investment),
                                ROS (return of spent)

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
        if "payment_rule" not in self.param_util:
            raise ValueError("specify payment rule")
        if "utility_type" not in self.param_util:
            self.param_util["utility_type"] = "QL"
            print("utility type not specified, quasi-linear (QL) chosen by default.")

        self.payment_rule = param_util["payment_rule"]
        self.tie_breaking = param_util["tie_breaking"]
        self.utility_type = param_util["utility_type"]

        # prior
        if self.prior == "affiliated_values":
            self.value_model = "affiliated"
        elif self.prior == "common_value":
            self.value_model = "common"
            self.v_space = {i: [0, o_space[i][1] / 2] for i in bidder}

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
        self.test_input_utility(obs, bids, idx)
        valuations = self.get_valuation(obs)

        allocation = self.get_allocation(bids, idx)
        payment = self.get_payment(bids, idx)
        payoff = self.get_payoff(valuations, allocation, payment)

        return payoff

    def get_allocation(self, bids: np.ndarray, idx: int) -> tuple:
        """compute allocation given action profiles

        Args:
            bids (np.ndarray): action profiles
            idx (int): index of agent we consider

        Returns:
            tuple: 2x np.ndarray, allocation vector for agent idx, number of ties
        """
        if self.tie_breaking == "random":
            is_winner = np.where(bids[idx] >= np.delete(bids, idx, 0).max(axis=0), 1, 0)
            num_winner = (bids.max(axis=0) == bids).sum(axis=0)

        elif self.tie_breaking == "lose":
            is_winner = np.where(bids[idx] > np.delete(bids, idx, 0).max(axis=0), 1, 0)
            num_winner = np.ones(is_winner.shape)
        else:
            raise ValueError(
                'Tie-breaking rule "' + self.param_util["tie_breaking"] + '" unknown'
            )
        allocation = is_winner / num_winner
        return allocation

    def get_payment(self, bids: np.ndarray, idx: int) -> np.ndarray:
        """compute payment (assuming bidder idx wins)

        Args:
            bids (np.ndarray): action profiles
            idx (int): index of agent we consider

        Returns:
            np.ndarray: payment vector
        """
        if self.payment_rule == "first_price":
            payment = bids[idx]
        elif self.payment_rule == "second_price":
            payment = np.delete(bids, idx, 0).max(axis=0)
        else:
            raise ValueError("payment rule " + self.payment_rule + " not available")
        return payment

    def get_payoff(
        self, valuation: np.ndarray, allocation: np.ndarray, payment: np.ndarray
    ) -> np.ndarray:
        """compute payoff given allocation and payment vector for different utility types:
        QL: quasi-linear, ROI: return of investement, ROS: return on spend, ROSB: return on spend with budget

        Args:
            valuation (np.ndarray) : valuation of bidder (idx), equal to observation in private value model
            allocation (np.ndarray): allocation vector for agent
            payment (np.ndarray): payment vector for agent ()

        Returns:
            np.ndarray: payoff
        """
        if self.utility_type == "QL":
            payoff = allocation * (valuation - payment)

        elif self.utility_type == "ROI":
            # if payment is zero, payoff is set to zero
            payoff = allocation * np.divide(
                valuation - payment,
                payment,
                out=np.zeros_like((allocation)),
                where=payment != 0,
            )
        elif self.utility_type == "ROS":
            payoff = allocation * np.divide(
                valuation,
                payment,
                out=np.zeros_like((valuation / np.ones(allocation.shape))),
                where=payment != 0,
            )
        elif self.utility_type == "ROSB":
            payoff = np.divide(
                valuation,
                payment,
                out=np.zeros_like((valuation / np.ones(allocation.shape))),
                where=payment != 0,
            ) + np.log(self.param_util["budget"] - payment)
        else:
            raise ValueError("utility type " + self.utility_type + " not available")
        return payoff

    def get_valuation(self, obs: np.ndarray, bids: np.ndarray, idx: int) -> np.ndarray:
        """determine valuations (potentially from observations, might be equal for private value model)
        and reformat vector depending on the use case:
            - one valuation for each action profile (no reformatting), needed for simulation
            - all valuations for each action profule (reformatting), needed for gradient computation (game.py)

        Args:
            obs (np.ndarray): observation of agent (idx)
            bids (np.ndarray): bids of all agents
            idx (int): index of agent

        Returns:
            np.ndarray: observations, possibly reformated
        """
        if (self.value_model == "private") or (self.value_model == "common"):
            if obs.shape != bids[idx].shape:
                valuations = obs.reshape(len(obs), 1)
            else:
                valuations = obs
        elif self.value_model == "affiliated":
            if obs[idx].shape != bids[idx].shape:
                valuations = 0.5 * (
                    obs.reshape(len(obs), 1) + obs.reshape(1, len(obs))
                ).reshape(len(obs), len(obs), 1)
            else:
                valuations = obs.mean(axis=0)
        else:
            raise NotImplementedError(f"value model {self.value_model} unknown")
        return valuations

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
            if (self.prior == "uniform") & (self.payment_rule == "first_price"):
                if np.all([self.o_space[i] == [0, 1] for i in self.set_bidder]):
                    if np.all([self.a_space[i] == [0, 1] for i in self.set_bidder]):
                        return (self.n_bidder - 1) / (self.n_bidder) * obs
                    elif np.all(
                        [
                            self.a_space[i][1] == 1
                            and self.a_space[i][0] > 0
                            and self.a_space[i][0] <= 1
                            for i in self.set_bidder
                        ]
                    ):
                        reserve_price = self.a_space[self.set_bidder[0]][0]
                        obs_clipped = np.clip(obs, reserve_price, None)
                        bne = ((self.n_bidder - 1) / self.n_bidder) * obs_clipped + (
                            1 / self.n_bidder
                        ) * (
                            reserve_price**self.n_bidder
                            / obs_clipped ** (self.n_bidder - 1)
                        )
                        return bne

            elif ((self.prior == "uniform") | (self.prior == "gaussian")) & (
                (self.payment_rule == "second_price")
            ):
                if np.all([self.o_space[i][0] == 0 for i in self.set_bidder]):
                    return obs
                elif np.all([self.a_space[i][0] > 0 for i in self.set_bidder]):
                    reserve_price = self.a_space[self.set_bidder[0]][0]
                    return np.clip(obs, reserve_price, None)

            elif self.prior == "affiliated_values":
                return 2 / 3 * obs

            elif self.prior == "common_value":
                return 2 * obs / (2 + obs)

        elif self.utility_type == "ROI":
            if (
                (self.prior == "uniform")
                & (self.payment_rule == "first_price")
                & (len(self.set_bidder) == 1)
                & (self.a_space[self.set_bidder[0]][0] > 0)
            ):
                reserve_price = self.a_space[self.set_bidder[0]][0]
                x = np.clip(obs, reserve_price, None)
                if self.n_bidder == 2:
                    return x / (-np.log(reserve_price) + np.log(x) + 1)

                elif self.n_bidder == 5:
                    return -3 * x**4 / (reserve_price**3 - 4 * x**3)

            elif ((self.prior == "uniform") | (self.prior == "gaussian")) & (
                (self.payment_rule == "second_price")
            ):
                if np.all([self.o_space[i][0] == 0 for i in self.set_bidder]):
                    return obs
                elif np.all([self.a_space[i][0] > 0 for i in self.set_bidder]):
                    reserve_price = self.a_space[self.set_bidder[0]][0]
                    return np.clip(obs, reserve_price, None)

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

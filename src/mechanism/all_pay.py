from typing import Dict, List

import numpy as np
from scipy.special import binom

from src.game import Game
from src.mechanism.mechanism import Mechanism
from src.strategy import Strategy
from src.util import mechanism_util

# -------------------------------------------------------------------------------------------------------------------- #
#                                                ALL-PAY AUCTION                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


class AllPayAuction(Mechanism):
    """All-pay Auction

    Parameter Mechanism
        bidder, o_space, a_space - standard input for all mechanism (see class Mechanism)


    Parameter Prior (param_prior)
        distribution


    Parameter Utility (param_util)
        tiebreaking     str: specifies tiebreaking rule: "random", "index" or "lose"

        type            str: agent's type can be interpreted as "valuation" or "cost"

        payment_rule    str:  choose between first_price, second_price and generalized (convex combination of both)
                        for the latter we need additional payment_param in param_util

        utility_type    str: choose between different risk-aversion models. If utility_type is not given it defaults to RN.
                        RN (risk neutral): U(x) = x, where x is the payoff
                        CRRA (constant relative risk aversion): U(x) = x^{1-eps}
    """

    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        param_prior: Dict[str, str],
        param_util: Dict[str, float],
    ):
        super().__init__(bidder, o_space, a_space, param_prior, param_util)
        self.name = "all_pay"

        self.check_param()
        self.tie_breaking = param_util["tie_breaking"]
        self.payment_rule = param_util["payment_rule"]
        self.utility_type = param_util["utility_type"]
        self.type = param_util["type"]

        self.check_own_gradient()

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int) -> np.ndarray:
        """Utility function for All-Pay Auction

        Args:
            obs (np.ndarray): valuation/skill parameter
            bids (np.ndarray): bid profiles
            idx (int): agent

        Returns:
            np.ndarray: payoff vector for agent
        """
        self.test_input_utility(obs, bids, idx)

        valuation = self.get_valuation(obs, bids, idx)
        allocation = self.get_allocation(bids, idx)
        payment = self.get_payment(bids, allocation, idx)
        payoff = self.get_payoff(valuation, allocation, payment)

        return payoff

    def get_payoff(
        self, allocation: np.ndarray, valuation: np.ndarray, payment: np.ndarray
    ) -> np.ndarray:
        """compute payoff given allocation and payment vector for different utility types:
        in particular we consider different types of risk-aversion

        Args:
            allocation (np.ndarray): allocation vector for agent
            valuation (np.ndarray) : valuation of bidder (idx), equal to observation in private value model
            payment (np.ndarray): payment vector for agent ()

        Returns:
            np.ndarray: payoff
        """
        payoff = allocation * valuation - payment
        if self.utility_type == "RN":
            return payoff
        elif self.utility_type == "CRRA":
            rho = self.param_util["risk_parameter"]
            return np.sign(payoff) * np.abs(payoff) ** rho
        else:
            raise ValueError("utility_type {self.utility_typ} unknown")

    def get_payment(
        self, bids: np.ndarray, allocation: np.ndarray, idx: int
    ) -> np.ndarray:
        """compute payment for bidder idx

        Args:
            bids (np.ndarray): action profiles
            allocation (np.ndarray): allocation vector for agent i
            idx (int): index of agent we consider

        Returns:
            np.ndarray: payment vector
        """
        if self.payment_rule == "first_price":
            return bids[idx]

        elif self.payment_rule == "second_price":
            second_price = np.delete(bids, idx, 0).max(axis=0)
            return allocation * second_price + (1 - allocation) * bids[idx]

        elif self.payment_rule == "generalized":
            alpha = self.param_util["payment_parameter"]
            second_price = np.delete(bids, idx, 0).max(axis=0)

            return (
                allocation * ((1 - alpha) * bids[idx] + alpha * second_price)
                + (1 - allocation) * bids[idx]
            )

        else:
            raise NotImplementedError(
                f"payment rule {self.payment_rule} not implemented"
            )

    def get_allocation(self, bids: np.ndarray, idx: int) -> np.ndarray:
        """compute allocation given action profiles for agent i

        Args:
            bids (np.ndarray): action profiles
            idx (int): index of agent we consider

        Returns:
            np.ndarray: allocation vector for agent idx
        """
        return mechanism_util.get_allocation_single_item(
            bids, idx, self.tie_breaking, zero_wins=True
        )

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
        bnes = [
            self.bne_uniform_first_price(agent, obs),
            self.bne_uniform_generalized(agent, obs),
            self.bne_powerlaw_first_price(agent, obs),
        ]
        for bne in bnes:
            if bne is not None:
                return bne
        return None

    def bne_uniform_first_price(self, agent: str, obs: np.ndarray):
        """BNE forall-pay auction with uniform prior, risk-neutral bidders, and first-price payment rule"""
        if (
            self.check_bidder_symmetric([0, 1])
            & (self.utility_type == "RN")
            & (self.payment_rule == "first_price")
        ):
            bne = (self.n_bidder - 1) / self.n_bidder * obs**self.n_bidder
            return bne
        else:
            return None

    def bne_uniform_generalized(self, agent: str, obs: np.ndarray):
        """BNE for all-pay auction with uniform prior, 2 risk-neutral bidders, and generalized payment rule"""
        if (
            self.check_bidder_symmetric([0, 1])
            & (self.prior == "uniform")
            & (self.utility_type == "RN")
            & (self.payment_rule == "generalized")
            & (self.n_bidder == 2)
        ):
            alpha = self.param_util["payment_parameter"]
            bne = -obs / alpha - 1 / alpha**2 * np.log(1 - alpha * obs)
            return bne
        else:
            return None

    def bne_powerlaw_first_price(self, agent: str, obs: np.ndarray):
        """BNE for all-pay auction with powerlaw prior, 2 risk-neutral bidders and first-price payment rule"""
        if (
            self.check_bidder_symmetric([0, 1])
            & (self.prior == "powerlaw")
            & (self.utility_type == "RN")
            & (self.payment_rule == "first_price")
            & (self.n_bidder == 2)
        ):
            power = self.param_prior["power"]
            bne = power / (power + 1) * obs ** (power + 1)
            return bne
        else:
            return None

    def compute_gradient(self, game: Game, strategies: Dict[str, Strategy], agent: str):
        """Simplified computation of gradient for i.i.d. bidders

        Parameters
        ----------
        strategies :
        game :
        agent :

        Returns
        -------

        """
        prob_win = mechanism_util.compute_probability_winning(game, strategies, agent)

        obs_grid = (
            np.ones((strategies[agent].m, strategies[agent].n))
            * strategies[agent].o_discr
        ).T
        bid_grid = (
            np.ones((strategies[agent].n, strategies[agent].m))
            * strategies[agent].a_discr
        )
        return prob_win * self.get_payoff(
            np.ones_like(obs_grid), obs_grid, bid_grid
        ) + (1 - prob_win) * self.get_payoff(
            np.zeros_like(obs_grid), obs_grid, bid_grid
        )

    def check_param(self):
        """
        Check if input paremter are sufficient to define mechanism
        """
        if "tie_breaking" not in self.param_util:
            raise ValueError("specify tiebreaking in param_util")

        if "type" not in self.param_util:
            raise ValueError("specify param_util - type: cost or valuation")

        if "utility_type" not in self.param_util:
            self.param_util["utility_type"] = "RN"
        else:
            if self.param_util["utility_type"] in ["CRRA"]:
                if "risk_parameter" not in self.param_util:
                    raise ValueError("Specify risk_parameter for CRRA")

        if "payment_rule" not in self.param_util:
            raise ValueError(
                "specify payment_rule in param_util - payment_rule: first_price, generalized, "
            )
        else:
            if self.param_util["payment_rule"] == "generalized":
                if self.n_bidder != 2:
                    raise ValueError(
                        "Generalized Payment rule only available for two bidders"
                    )
                if "payment_parameter" not in self.param_util:
                    raise ValueError(
                        "Specify payment_parameter in param_util for generalized payment_rule"
                    )
                else:
                    if (self.param_util["payment_parameter"] < 0) or (
                        self.param_util["payment_parameter"] > 1
                    ):
                        raise ValueError("payment_param has to be between 0 and 1")

    def check_own_gradient(self):
        """check if we can use gradient computation of mechanism"""
        if (
            self.check_bidder_symmetric([0, 1])
            & ("corr" not in self.param_prior)
            & (self.payment_rule == "first_price")
            & (self.type == "valuation")
            & (self.tie_breaking == "lose")
        ):
            self.own_gradient = True
            # print("- gradient computation via mechanism -")

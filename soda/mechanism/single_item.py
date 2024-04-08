from typing import Dict, List

import numpy as np

from soda.game import Game
from soda.mechanism.mechanism import Mechanism
from soda.mechanism.util import (
    compute_probability_order,
    compute_probability_winning,
    get_allocation_single_item,
)
from soda.strategy import Strategy

# -------------------------------------------------------------------------------------------------------------------- #
#                                             SINGLE-ITEM AUCTION                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


class SingleItemAuction(Mechanism):
    """Single-Item Auction
    We assume that bidders cannot win with zero bids

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
        reserve_price   float: specifies minimal payment if agent wins. Defaults to 0.
                        Note that we reserve_price only affect the pricing rule, not the allocation (i.e., you can still win with lower bids)

    """

    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        param_prior: Dict,
        param_util: Dict,
    ):
        super().__init__(bidder, o_space, a_space, param_prior, param_util)
        self.name = "single_item"

        self.check_param()
        self.payment_rule = self.param_util["payment_rule"]
        self.tie_breaking = self.param_util["tie_breaking"]
        self.utility_type = self.param_util["utility_type"]
        self.reserve_price = self.param_util["reserve_price"]

        self.check_own_gradient()

        # prior
        if self.prior == "affiliated_values":
            self.value_model = "common_affiliated"
        elif self.prior == "common_value":
            self.value_model = "common_independent"
            self.v_space = [0, o_space[bidder[0]][1] / 2]

    # ------------------------------- methods to compute utilities --------------------------------- #

    def utility(
        self, obs_profile: np.ndarray, bids_profile: np.ndarray, index_agent: int
    ) -> None:
        """Compute utility for agent in single_item auction

        Args:
            obs_profile (np.ndarray): observations of all agents
            bids_profile (np.ndarray): bids of all agents
            index_agent (int): index of agent

        Returns:
            np.ndarry: utilities of agent (with index index_agent)
        """
        self.test_input_utility(obs_profile, bids_profile, index_agent)
        valuation = self.get_valuation(obs_profile, index_agent)

        allocation = self.get_allocation(bids_profile, index_agent)
        payment = self.get_payment(bids_profile, allocation, index_agent)
        payoff = self.get_payoff(
            allocation=allocation, valuation=valuation, payment=payment
        )
        return payoff

    def revenue(self, bid_profile: np.ndarray) -> np.ndarray:
        """Compute revenue (sum of payments) for auctioneer

        Args:
            bid_profile (np.ndarray): bids of all agents

        Returns:
            np.ndarray: revenues
        """
        allocations = np.array(
            [self.get_allocation(bid_profile, i) for i in range(self.n_bidder)]
        )
        payments = np.array(
            [
                self.get_payment(bid_profile, allocations[i], i)
                for i in range(self.n_bidder)
            ]
        )
        revenue = (allocations * payments).sum(axis=0)
        return revenue

    def get_payoff(
        self, allocation: np.ndarray, valuation: np.ndarray, payment: np.ndarray
    ) -> np.ndarray:
        """compute payoff given allocation and payment vector for different utility types:
        QL: quasi-linear, ROI: return of investement, ROS: return on spend, ROSB: return on spend with budget

        Args:
            allocation (np.ndarray): allocation vector for agent
            valuation (np.ndarray) : valuation of bidder (idx), equal to observation in private value model
            payment (np.ndarray): payment vector for agent ()

        Returns:
            np.ndarray: payoff
        """
        if self.utility_type == "QL":
            payoff = valuation - payment

        elif self.utility_type == "ROI":
            # if payment is zero, payoff is set to zero
            payoff = np.divide(
                valuation - payment,
                payment,
                out=np.zeros_like((valuation - payment)),
                where=payment != 0,
            )
        elif self.utility_type == "ROS":
            payoff = np.divide(
                valuation,
                payment,
                out=np.zeros_like((valuation - payment)),
                where=payment != 0,
            )
        elif self.utility_type == "ROSB":
            payoff = np.divide(
                valuation,
                payment,
                out=np.zeros_like((valuation - payment)),
                where=payment != 0,
            ) + np.log(self.param_util["budget"] - payment)

        elif self.utility_type == "CRRA":
            rho = self.param_util["risk_parameter"]
            payoff = np.sign(valuation - payment) * np.abs(valuation - payment) ** rho

        else:
            raise ValueError(f"utility type {self.utility_type} not available")
        return allocation * payoff

    def get_payment(
        self, bids: np.ndarray, allocation: np.ndarray, idx: int
    ) -> np.ndarray:
        """compute payment
        we do not consider tie-breaking rules, if allocation > 0, full payment is computed
        payment is zero if agent does not win the item and it is half

        Args:
            bids (np.ndarray): action profiles
            alloaction (np.ndarray): allocation vector for agent idx
            idx (int): index of agent we consider

        Returns:
            np.ndarray: payment vector
        """
        if self.payment_rule == "first_price":
            payment = np.clip(bids[idx], self.reserve_price, None)

        elif self.payment_rule == "second_price":
            payment = np.clip(
                np.delete(bids, idx, 0).max(axis=0), self.reserve_price, None
            )

        elif self.payment_rule == "third_price":
            payment = np.clip(
                np.sort(np.delete(bids, idx, 0), axis=0)[-2, :],
                self.reserve_price,
                None,
            )
        else:
            raise ValueError("payment rule " + self.payment_rule + " not available")
        return payment * np.where(allocation > 0, 1, 0)

    def get_allocation(self, bids: np.ndarray, idx: int) -> np.ndarray:
        """compute allocation given action profiles
        you cannot win with zero bids

        Args:
            bids (np.ndarray): action profiles
            idx (int): index of agent we consider

        Returns:
            np.ndarray: allocation vector for agent idx
        """
        return get_allocation_single_item(bids, idx, self.tie_breaking)

    def compute_valuations_from_observations(
        self, obs_profile: np.ndarray
    ) -> np.ndarray:
        """For the common values with correlated observation case (see mechanism.get_valuation()) we need a
        method to compute the valuations from the observation profile.

        Args:
            obs_profile (np.ndarray): _description_

        Returns:
            np.ndarray: common valuation
        """
        if (self.prior == "affiliated_values") and (self.n_bidder == 2):
            # affiliated values model example from Krishna
            return obs_profile.mean(axis=0)
        else:
            raise NotImplementedError

    # --------------------------------- methods to compute metrics --------------------------------- #

    def get_metrics(
        self, agent: str, obs_profile: np.ndarray, bid_profile: np.ndarray
    ) -> tuple:
        """compute all metrics relevant for single-item auction
        this includes standard metrics (l2, util_loss) + revenue
        """
        metrics, values = self.get_standard_metrics(agent, obs_profile, bid_profile)
        metrics += ["revenue"]
        values += [self.compute_expected_revenue(bid_profile)]

        return metrics, values

    def compute_expected_revenue(self, bid_profile: np.ndarray) -> float:
        """Computed expected revenue of single-item auction

        Args:
            bid_profile (np.ndarray): bid profile

        Returns:
            float: approximated expected revenue
        """
        allocations = np.array(
            [self.get_allocation(bid_profile, i) for i in range(self.n_bidder)]
        )
        payments = np.array(
            [
                self.get_payment(bid_profile, allocations[i], i)
                for i in range(self.n_bidder)
            ]
        )
        revenue = (allocations * payments).sum(axis=0)
        return revenue.mean()

    # -------------------------- methods to get equilbrium strategies------------------------------- #

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
            self.bne_ql_first_price(agent, obs),
            self.bne_ql_third_price(agent, obs),
            self.bne_second_price(agent, obs),
            self.bne_ql_first_price_affiliated(agent, obs),
            self.bne_ql_second_price_common(agent, obs),
            self.bne_roi_first_price(agent, obs),
            self.bne_crra_first_price(agent, obs),
        ]
        for bne in bnes:
            if bne is not None:
                return bne
        return None

    def bne_ql_first_price(self, agent: str, obs: np.ndarray):
        """BNE for first-price auction for quasi-linear utilities"""
        if (
            (self.utility_type == "QL")
            & (self.payment_rule == "first_price")
            & (self.prior == "uniform")
            & self.check_bidder_symmetric()
            & (self.value_model == "private")
        ):

            obs_clipped = np.clip(obs, self.reserve_price, None)
            bne = np.where(
                obs >= self.reserve_price,
                ((self.n_bidder - 1) / self.n_bidder) * obs_clipped
                + (1 / self.n_bidder)
                * (
                    self.reserve_price**self.n_bidder
                    / obs_clipped ** (self.n_bidder - 1)
                ),
                0,
            )
        else:
            bne = None
        return bne

    def bne_ql_first_price_affiliated(self, agent: str, obs: np.ndarray):
        """BNE for first-price auction for quasi-linear utilities with affiliated values"""
        if (
            (self.utility_type == "QL")
            & (self.payment_rule == "first_price")
            & self.check_bidder_symmetric([0, 2])
            & (self.value_model == "common_affiliated")
            & (self.n_bidder == 2)
        ):
            bne = 2 / 3 * obs
        else:
            bne = None
        return bne

    def bne_ql_third_price(self, agent: str, obs: np.ndarray):
        """BNE for third-price auction with quasi-linear utilities (IPV)"""
        if (
            (self.utility_type == "QL")
            & (self.payment_rule == "third_price")
            & self.check_bidder_symmetric([0, 1])
            & (self.prior == "uniform")
            & (self.value_model == "private")
            & (self.n_bidder >= 3)
        ):
            bne = obs + 1 / (self.n_bidder - 2) * obs
        else:
            bne = None
        return bne

    def bne_second_price(self, agent: str, obs: np.ndarray):
        """BNE for second-price auction for quasi-linear utilities (IPV)"""
        if (
            (self.utility_type in ["QL", "ROI"])
            & (self.payment_rule == "second_price")
            & self.check_bidder_symmetric()
            & (self.value_model == "private")
        ):
            bne = np.where(obs >= self.reserve_price, obs, 0)
        else:
            bne = None
        return bne

    def bne_ql_second_price_common(self, agent: str, obs: np.ndarray):
        """BNE for second-price auction for quasi-linear utilities with common"""
        if (
            (self.utility_type == "QL")
            & (self.payment_rule == "second_price")
            & self.check_bidder_symmetric([0, 2])
            & (self.value_model == "common_independent")
            & (self.n_bidder == 3)
        ):
            bne = 2 * obs / (2 + obs)
        else:
            bne = None
        return bne

    def bne_roi_first_price(self, agent: str, obs: np.ndarray):
        """BNE for first-price auction for resturn of invest maximizing agents"""
        if (
            (self.utility_type == "ROI")
            & (self.payment_rule == "first_price")
            & self.check_bidder_symmetric([0, 1])
            & (self.value_model == "private")
            & (self.prior == "uniform")
            & (self.reserve_price > 0)
        ):
            obs_clipped = np.clip(obs, self.reserve_price, None)
            if self.n_bidder == 2:
                bne = np.where(
                    obs >= self.reserve_price,
                    obs_clipped
                    / (-np.log(self.reserve_price) + np.log(obs_clipped) + 1),
                    0,
                )
            else:
                bne = np.where(
                    obs >= self.reserve_price,
                    ((self.n_bidder - 2) * obs_clipped ** (self.n_bidder - 1))
                    / (
                        (self.n_bidder - 1) * (obs_clipped ** (self.n_bidder - 2))
                        - self.reserve_price ** (self.n_bidder - 2)
                    ),
                    0,
                )
        else:
            bne = None
        return bne

    def bne_crra_first_price(self, agent: str, obs: np.ndarray):
        """BNE for first-price auction with constant relative risk averse bidders (CRRA)"""
        if (
            (self.utility_type == "CRRA")
            & (self.payment_rule == "first_price")
            & self.check_bidder_symmetric([0, 1])
            & (self.value_model == "private")
            & (self.prior == "uniform")
        ):
            rho = self.param_util["risk_parameter"]
            bne = obs * (self.n_bidder - 1) / (self.n_bidder - 1 + rho)
        else:
            bne = None
        return bne

    def bne_ql_third_price_uniform(self, agent: str, obs: np.ndarray):
        """BNE for first-price auction with constant relative risk averse bidders (CRRA)"""
        if (
            (self.utility_type == "QL")
            & (self.payment_rule == "third_price")
            & self.check_bidder_symmetric([0, 1])
            & (self.value_model == "private")
            & (self.prior == "uniform")
        ):
            bne = obs + 1 / (self.n_bidder - 2) * obs
        else:
            bne = None
        return bne

    # -------------------------- methods for faster gradient computation ------------------------------- #

    def compute_gradient(
        self, game: Game, strategies: Dict[str, Strategy], agent: str
    ) -> np.ndarray:
        """faster computation of gradient, ONLY FOR SYMMETRIC IID BIDDERS

        Args:
            game (Game): approximation game
            strategies (Dict[str, Strategy]): strategy profile
            agent (str): agent

        Returns:
            np.ndarray: gradient for agent
        """
        value_grid = (
            np.ones((strategies[agent].m, strategies[agent].n))
            * strategies[agent].o_discr
        ).T

        payment_grid = np.clip(
            np.ones((strategies[agent].n, strategies[agent].m))
            * strategies[agent].a_discr,
            self.reserve_price,
            None,
        )
        allocation_grid = np.ones_like(payment_grid)

        if self.payment_rule == "first_price":
            prob_win = compute_probability_winning(game, strategies, agent)
            payoff = self.get_payoff(allocation_grid, value_grid, payment_grid)
            return prob_win * payoff

        elif self.payment_rule == "second_price":
            pdf_order = compute_probability_order(game, strategies, agent)
            payoff = game.mechanism.get_payoff(
                allocation_grid, value_grid, payment_grid
            )
            prob_grid = np.ones((strategies[agent].n, strategies[agent].m)) * pdf_order
            return np.hstack([np.zeros((strategies[agent].n, 1)), payoff * prob_grid])[
                :, :-1
            ].cumsum(axis=1)

        else:
            raise ValueError(
                f"own gradient not available for payment_rule {self.payment_rule}"
            )

    def check_own_gradient(self):
        """check if we can use gradient computation of mechanism"""
        if (len(self.set_bidder) == 1) and (
            self.payment_rule in ["first_price", "second_price"]
        ) & (self.prior not in ["affiliated_values", "common_value"]) & (
            "corr" not in self.param_prior
        ):
            if self.payment_rule == "first_price":
                self.own_gradient = True
            elif (self.payment_rule == "second_price") & (self.tie_breaking == "lose"):
                self.own_gradient = True
            else:
                self.own_gradient = False

    # -------------------------------------- helper methods --------------------------------------------- #

    def check_param(self):
        """
        Check if input paremter are sufficient to define mechanism
        """
        if "tie_breaking" not in self.param_util:
            raise ValueError("specify tiebreaking rule")
        if "payment_rule" not in self.param_util:
            raise ValueError("specify payment rule")
        else:
            if (self.param_util["payment_rule"] == "third_price") & (self.n_bidder < 3):
                raise ValueError(
                    "Third-price payment rule only available for more than 2 agents"
                )
        if "utility_type" not in self.param_util:
            self.param_util["utility_type"] = "QL"
            print("utility_type not specified, quasi-linear (QL) chosen by default.")
        else:
            if self.param_util["utility_type"] == "CRRA":
                if "risk_parameter" not in self.param_util:
                    raise ValueError(
                        "Specify risk_parameter if using utility_type: CRRA"
                    )
        if "reserve_price" not in self.param_util:
            self.param_util["reserve_price"] = 0.0

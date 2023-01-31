from typing import Dict, List

import numpy as np
from scipy.special import binom

from src.game import Game
from src.mechanism.mechanism import Mechanism
from src.strategy import Strategy

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
            self.value_model = "affiliated"
        elif self.prior == "common_value":
            self.value_model = "common"
            self.v_space = {i: [0, o_space[i][1] / 2] for i in bidder}

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int) -> None:
        """Utility function for Single-Item Auction

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
        valuation = self.get_valuation(obs, bids, idx)

        allocation = self.get_allocation(bids, idx)
        payment = self.get_payment(bids, idx)
        payoff = self.get_payoff(valuation, allocation, payment)

        return payoff

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
            payoff = allocation * valuation - payment

        elif self.utility_type == "ROI":
            # if payment is zero, payoff is set to zero
            payoff = np.divide(
                allocation * valuation - payment,
                payment,
                out=np.zeros_like((valuation - payment)),
                where=payment != 0,
            )
        elif self.utility_type == "ROS":
            payoff = np.divide(
                allocation * valuation,
                payment,
                out=np.zeros_like((valuation - payment)),
                where=payment != 0,
            )
        elif self.utility_type == "ROSB":
            payoff = np.divide(
                allocation * valuation,
                payment,
                out=np.zeros_like((valuation - payment)),
                where=payment != 0,
            ) + np.log(self.param_util["budget"] - payment)
        else:
            raise ValueError("utility type " + self.utility_type + " not available")
        return payoff

    def get_payment(
        self, bids: np.ndarray, allocation: np.ndarray, idx: int
    ) -> np.ndarray:
        """compute payment (assuming bidder idx wins)

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
        return payment * np.where(allocation > 0, 1.0, 0.0)

    def get_allocation(self, bids: np.ndarray, idx: int) -> np.ndarray:
        """compute allocation given action profiles
        you cannot win with zero bids

        Args:
            bids (np.ndarray): action profiles
            idx (int): index of agent we consider

        Returns:
            np.ndarray: allocation vector for agent idx
        """
        if self.tie_breaking == "random":
            is_winner = np.where(
                (bids[idx] >= np.delete(bids, idx, 0).max(axis=0)) & (bids[idx] > 0),
                1,
                0,
            )
            num_winner = (bids.max(axis=0) == bids).sum(axis=0)

        elif self.tie_breaking == "lose":
            is_winner = np.where(bids[idx] > np.delete(bids, idx, 0).max(axis=0), 1, 0)
            num_winner = np.ones(is_winner.shape)
        else:
            raise ValueError(
                'Tie-breaking rule "' + self.param_util["tie_breaking"] + '" unknown'
            )
        return is_winner / num_winner

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
        setting_fits, bne = self.bne_ql_first_price(agent, obs)
        if setting_fits:
            return bne

        setting_fits, bne = self.bne_second_price(agent, obs)
        if setting_fits:
            return bne

        setting_fits, bne = self.bne_ql_first_price_affiliated(agent, obs)
        if setting_fits:
            return bne

        setting_fits, bne = self.bne_ql_second_price_common(agent, obs)
        if setting_fits:
            return bne

        setting_fits, bne = self.bne_roi_first_price(agent, obs)
        if setting_fits:
            return bne

        return None

    def bne_ql_first_price(self, agent: str, obs: np.ndarray):
        """BNE for first-price auction for quasi-linear utilities"""
        # check if setting applies
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
            return True, bne
        else:
            return False, None

    def bne_ql_first_price_affiliated(self, agent: str, obs: np.ndarray):
        """BNE for first-price auction for quasi-linear utilities with affiliated values"""
        if (
            (self.utility_type == "QL")
            & (self.payment_rule == "first_price")
            & self.check_bidder_symmetric([0, 2])
            & (self.value_model == "affiliated")
            & (self.n_bidder == 2)
        ):
            bne = 2 / 3 * obs
            return True, bne
        else:
            return False, None

    def bne_second_price(self, agent: str, obs: np.ndarray):
        """BNE for second-price auction for quasi-linear utilities (IPV)"""
        if (
            (self.utility_type in ["QL", "ROI"])
            & (self.payment_rule == "second_price")
            & self.check_bidder_symmetric()
            & (self.value_model == "private")
        ):
            bne = np.where(obs >= self.reserve_price, obs, 0)
            return True, bne
        else:
            return False, None

    def bne_ql_second_price_common(self, agent: str, obs: np.ndarray):
        """BNE for second-price auction for quasi-linear utilities with common"""
        if (
            (self.utility_type == "QL")
            & (self.payment_rule == "second_price")
            & self.check_bidder_symmetric([0, 2])
            & (self.value_model == "common")
            & (self.n_bidder == 3)
        ):
            bne = 2 * obs / (2 + obs)
            return True, bne
        else:
            return False, None

    def bne_roi_first_price(self, agent: str, obs: np.ndarray):
        """BNE for second-price auction for quasi-linear utilities with affiliated values"""
        if (
            (self.utility_type == "ROI")
            & (self.payment_rule == "first_price")
            & self.check_bidder_symmetric([0, 1])
            & (self.value_model == "private")
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
            return True, bne
        else:
            return False, None

    def compute_gradient(self, game: Game, strategies: Dict[str, Strategy], agent: str):
        """Simplified computation of gradient for i.i.d. bidders and tie-breaking "lose"

        Parameters
        ----------
        strategies :
        game :
        agent :

        Returns
        -------

        """
        prob_win = self.compute_probability_winning(game, strategies, agent)

        # utility type
        obs_grid = (
            np.ones((strategies[agent].m, strategies[agent].n))
            * strategies[agent].o_discr
        ).T
        # first-price (including reserve price)
        bid_grid = np.clip(
            np.ones((strategies[agent].n, strategies[agent].m))
            * strategies[agent].a_discr,
            self.reserve_price,
            None,
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

        return prob_win * payoff

    def compute_probability_winning(
        self, game: Game, strategies: Dict[str, Strategy], agent: str
    ) -> np.ndarray:
        """Compute probability of winning for each bid given the symmetric (!) strategies

        Args:
            game (Game): approximation game
            strategies (Dict[str, Strategy]): strategy profile
            agent (str): bidder

        Raises:
            ValueError: wront tie breaking rule

        Returns:
            np.ndarray: probability of winning
        """

        pdf = strategies[agent].x.sum(axis=0)
        cdf = np.insert(pdf, 0, 0.0).cumsum()[:-1]
        prob_win = cdf ** (self.n_bidder - 1)

        if self.tie_breaking == "lose":
            pass
        elif self.tie_breaking == "random":
            prob_win += sum(
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

        return prob_win

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
        if "reserve_price" not in self.param_util:
            self.param_util["reserve_price"] = 0.0

    def check_own_gradient(self):
        """check if we can use gradient computation of mechanism"""
        if (len(self.set_bidder) == 1) and (self.payment_rule == "first_price") & (
            self.prior not in ["affiliated_values", "common_value"]
        ) & ("corr" not in self.param_prior):
            self.own_gradient = True
            # print("- gradient computation via mechanism -")

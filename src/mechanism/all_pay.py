from typing import Dict, List

import numpy as np

from .mechanism import Mechanism

# -------------------------------------------------------------------------------------------------------------------- #
#                                                ALL-PAY AUCTION                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


class AllPay(Mechanism):
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
        self.type = param_util["type"]

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

        return allocation * valuation - payment

    def get_allocation(self, bids: np.ndarray, idx: int) -> np.ndarray:
        """compute allocation given action profiles for agent i

        Args:
            bids (np.ndarray): action profiles
            idx (int): index of agent we consider

        Returns:
            np.ndarray: allocation vector for agent idx
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
        return is_winner / num_winner

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

        if (self.n_bidder == 2) & self.check_bidder_symmetric:

            if (self.prior == "uniform") & (self.o_space[self.bidder[0]] == [0, 1]):
                if self.payment_rule == "first_price":
                    return 1 / 2 * obs**2
                elif self.payment_rule == "generalized":
                    alpha = self.param_util["payment_parameter"]
                    return -obs / self.param_allpay - 1 / self.param_allpay**2 * np.log(
                        1 - self.param_allpay * obs
                    )
                else:
                    return None

            elif (self.prior == "powerlaw") & (self.o_space[self.bidder[0]] == [0, 1]):
                power = self.param_prior["power"]
                return power / (power + 1) * obs ** (power + 1)

            else:
                raise NotImplemented(
                    "BNE not implemented for this setting (prior, spaces)"
                )
        else:
            raise NotImplemented("BNE not implemented for more than 2 agents")

    def check_param(self):
        """
        Check if input paremter are sufficient to define mechanism
        """
        if "tie_breaking" not in self.param_util:
            raise ValueError("specify tiebreaking in param_util")
        if "type" not in self.param_util:
            raise ValueError("specify param_util - type: cost or valuation")
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
                    if (self.param_util["payment_param"] < 0) or (
                        self.param_util["payment_parameter"] > 1
                    ):
                        raise ValueError("payment_param has to be between 0 and 1")

from typing import Dict, List

import numpy as np

from .mechanism import Mechanism

# -------------------------------------------------------------------------------------------------------------------- #
#                                               SPLIT AWARD AUCTION                                                    #
# -------------------------------------------------------------------------------------------------------------------- #


class SplitAwardAuction(Mechanism):
    """Split-Award Auction
    The split award auction is a procurement auction with two bidders. They can bid on a split (50%) or sole source (100%).

    Parameter Mechanism
        bidder, o_space, a_space - standard input for all mechanism (see class Mechanism)


    Parameter Prior (param_prior)
        distribution    str: e.g. uniform, gaussian_trunc


    Parameter Utility (param_util)
        tiebreaking     str: specifies tiebreaking rule: "random" (default), "lose"
        payment_rule    str: choose betweem "first_price" and "second_price"
        utility_type    str:    QL (quasi-linear (corresponds to Auction, Default),
                                ROI (return of investment),
                                ROS (return of spent)
        scale           float: factor for diseconomies of scale

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
        self.name = "split_award"

        self.check_param()
        self.payment_rule = param_util["payment_rule"]
        self.tie_breaking = param_util["tie_breaking"]
        self.scale = param_util["scale"]

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int):
        """Compute Utility for Split Award Auction

        Args:
            obs (np.ndarray): observations/valuations from each agent
            bids (np.ndarray): bids (2 dim: single, split) from each agent
            idx (int): index of agent

        """
        self.test_input_utility(obs, bids, idx)
        valuation = self.get_valuation(obs, bids, idx)

        allocation = self.get_allocation(bids, idx)
        payment = self.get_payment(bids, idx)

        return allocation["single"] * (payment["single"] - valuation) + allocation[
            "split"
        ] * (payment["split"] - self.scale * valuation)

    def get_allocation(self, bids: np.ndarray, idx: int) -> np.ndarray:
        """compute allocation given action profiles
        split is preferred over single

        Args:
            bids (np.ndarray): action profiles
            idx (int): index of agent we consider

        Returns:
            np.ndarray: allocation vector for agent idx
        """
        idx_single, idx_split = 0, 1
        bids_single = np.array([bids[i][idx_single] for i in range(self.n_bidder)])
        bids_split = np.array([bids[i][idx_split] for i in range(self.n_bidder)])

        is_winner_split = np.where(
            bids_split.sum(axis=0) <= bids_single.min(axis=0), 1, 0
        )

        if self.param_util["tie_breaking"] == "random":
            is_winner_single = np.where(
                bids_single[idx] <= np.delete(bids_single, idx, 0).min(axis=0), 1, 0
            )
            num_winner_single = (bids_single[idx] == bids_single).sum(axis=0)

        elif self.param_util["tie_breaking"] == "lose":
            is_winner_single = np.where(
                bids_single[idx] < np.delete(bids_single, idx, 0).min(axis=0), 1, 0
            )
            num_winner_single = np.ones_like(is_winner_single)
        else:
            raise NotImplementedError(
                'tie-breaking rule "{}" not implemented'.format(
                    self.param_util["tie_breaking"]
                )
            )
        return {
            "single": is_winner_single / num_winner_single * (1 - is_winner_split),
            "split": is_winner_split,
        }

    def get_payment(self, bids: np.ndarray, idx: int) -> np.ndarray:
        """compute payment (assuming bidder idx wins)

        Args:
            bids (np.ndarray): action profiles
            idx (int): index of agent we consider

        Returns:
            np.ndarray: payment vector
        """
        if self.payment_rule == "first_price":
            return {"single": bids[idx][0], "split": bids[idx][1]}
        elif self.payment_rule == "second_price":
            return {"single": bids[1 - idx][0], "split": bids[1 - idx][1]}
        else:
            raise ValueError("payment rule " + self.payment_rule + " not available")

    def get_valuation(self, obs: np.ndarray, bids: np.ndarray, idx: int) -> np.ndarray:
        """determine valuations (potentially from observations, might be equal for private value model)
        and reformat vector depending on the use case (specific for split-award)
            - one valuation for each action profile (no reformatting), needed for simulation
            - all valuations for each action profule (reformatting), needed for gradient computation (game.py)

        Args:
            obs (np.ndarray): observation of agent (idx)
            bids (np.ndarray): bids of all agents
            idx (int): index of agent

        Returns:
            np.ndarray: observations, possibly reformated
        """
        if self.value_model == "private":
            if obs.shape != bids[idx][0].shape:
                valuations = obs.reshape(len(obs), 1)
            else:
                valuations = obs
        else:
            raise NotImplementedError(
                "get valuation only implemented for the private value model"
            )
        return valuations

    def get_bne(self, agent: str, obs: np.ndarray):
        return self.equilibrium_pooling(agent, obs, bound="upper")

    def equilibrium_pooling(
        self, agent: str, obs: np.ndarray, bound: str = "upper"
    ) -> np.ndarray:
        """Pooling-Equilibrium in standard setting. There is a continuum of equilibria.

        Args:
            agent (str): bidder
            obs (np.ndarray): observation
            bound (str, optional): upper or lower bound of equilibriums continuum

        Returns:
            np.ndarray: pooling_equilibrium (single, split)
        """
        if bound == "upper":
            split_bid_max = (1 - self.scale) * self.o_space[agent][0]

            return np.array(
                [
                    split_bid_max
                    + (
                        split_bid_max
                        - self.o_space[agent][1]
                        * self.scale
                        * (obs - self.o_space[agent][0])
                        / (self.o_space[agent][1] - self.o_space[agent][0])
                    )
                    / (
                        1
                        - (obs - self.o_space[agent][0])
                        / (self.o_space[agent][1] - self.o_space[agent][0])
                    ),
                    split_bid_max * np.ones(obs.shape),
                ]
            )
        elif bound == "lower":
            return np.array(
                [
                    2 * self.scale * self.o_space[agent][1] * np.ones(len(obs)),
                    self.scale * self.o_space[agent][1] * np.ones(len(obs)),
                ]
            )
        else:
            raise ValueError('Choose "upper" or "lower" bound for pooling equilibria')

    def equilibrium_wta(self, agent: str, obs: np.ndarray) -> np.ndarray:
        """WTA (Winner-Takes-All) Equilibrium in standard setting

        Args:
            agent (str): bidder
            obs (np.ndarray): observation

        Returns:
            np.ndarray: wta_equilibrium (single, split)
        """

        return np.array(
            [
                0.5 * (obs + self.o_space[agent][1]),
                0.5 * (obs + self.o_space[agent][1])
                - self.scale * self.o_space[agent][0],
            ]
        )

    def check_param(self):
        """check if input parameter are sufficient to define mechanism"""
        if self.n_bidder != 2:
            raise NotImplementedError(
                "split-award mechanism only implemented for 2 bidders"
            )
        if "tie_breaking" not in self.param_util:
            raise ValueError("specify tie_breaking in param_util")
        if "payment_rule" not in self.param_util:
            raise ValueError("specify payment_rule in param_util")
        if "scale" not in self.param_util:
            raise ValueError("specify scale in param_util")

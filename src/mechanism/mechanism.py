from typing import Dict, List

import numpy as np
from scipy.stats import norm, powerlaw, truncnorm, uniform

# -------------------------------------------------------------------------------------------------------------------- #
#                                         MECHANISM : CONTINUOUS AUCTION GAME                                          #
# -------------------------------------------------------------------------------------------------------------------- #


class Mechanism:
    """Mechanism represents the underlying continuous Bayesian (auction) game

    Attributes:
        General
            bidder          List: contains all agents (str)
            set_bidder      List: contains all unique agents (model sharing)
            o_space         Dict: contains limits of observation space [a,b] for agents
            a_space         Dict: contains limits of action space [a,b] for agents
            param_prior     Dict: contains parameters for prior, e.g., distribution and parameter

        Speficiations of Mechanism
            own_gradient    boolean: if true mechanism contains own function to compute gradient
            value_model     str: private, common, affiliated

    Methods:
        utility: abstract method to compute utilities, uses the following methods
            - test_input_utility: test input to compute utility
            - get_valuation: reformates observations/valuations and consideres different value models
            - get_allocation, get_payment, get_payoff are used as well but implement in the respective child classes

        sample_types: draws types (valuations/observations) according to given prior. The following priors are implemented
            - uniform, gaussian, gaussian_trunc, powerlaw

    """

    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        param_prior: Dict,
        param_util: Dict,
    ):
        # bidder
        self.bidder = bidder
        self.n_bidder = len(bidder)
        self.set_bidder = list(set(bidder))

        # type and action space
        self.o_space = o_space
        self.a_space = a_space

        self.dim_o, self.dim_a = self.get_dimension_spaces()

        # param_prior
        self.param_prior = param_prior
        self.prior = param_prior["distribution"]

        # param_util
        self.param_util = param_util

        # further specifications
        self.name = None
        self.own_gradient = False
        self.value_model = "private"  # valuation depends only on own observation

    def __repr__(self) -> str:
        return f"Mechanism({self.name})"

    # ------------------------------- methods for computation of utilities ------------------------------------- #

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int):
        """Compute utility according to specified mechanism

        Args:
            obs (np.ndarray): observation of agent (idx)
            bids (np.ndarray): bids of all agents
            idx (int): index of agent

        Returns:
            np.ndarry: utilities of agend (idx)
        """
        raise NotImplementedError

    def get_allocation(self, bids: np.ndarray, idx: int) -> np.ndarray:
        """Compute alloction for idx-th bidder

        Args:
            bids (np.ndarray): action profiles
            idx (int): idx-th agent

        Returns:
            np.ndarray: allocation vector
        """
        raise NotImplementedError

    def get_payment(
        self, bids: np.ndarray, allocation: np.ndarray, idx: int
    ) -> np.ndarray:
        """Compute payments for idx-th bidder

        Args:
            bids (np.ndarray): action profiles
            allocation (np.ndarray): allocation for idx-th agent
            idx (int): idx-th agent

        Returns:
            np.ndarray: payment vector
        """
        raise NotImplementedError

    def test_input_utility(self, obs: np.ndarray, bids: np.ndarray, idx: int):
        if bids.shape[0] != self.n_bidder:
            raise ValueError("wrong format of bids")
        elif idx >= self.n_bidder:
            raise ValueError("bidder with index " + str(idx) + " not avaible")
        pass

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
        if self.value_model == "private":
            if obs.shape != bids[idx].shape:
                valuations = obs.reshape(len(obs), 1)
            else:
                valuations = obs
        else:
            raise NotImplementedError(
                "get valuation only implemented for the private value model"
            )
        return valuations

    # ---------------------------------- methods to compute metrics --------------------------------------- #

    def get_metrics(
        self, agent: str, obs_profile: np.ndarray, bid_profile: np.ndarray
    ) -> tuple:
        """Method for each mechanism to compute the metrics we are interest in. Has to be implemented in each mechanism"""
        raise NotImplementedError(
            f"get_metrics not implemented in {self.name} mechanism"
        )

    def get_standard_metrics(
        self, agent: str, obs_profile: np.ndarray, bid_profile: np.ndarray
    ) -> tuple:

        idx = self.bidder.index(agent)
        l2_norm = self.compute_l2_norm(agent, obs_profile[idx], bid_profile[idx])
        util_loss, util_vs_bne, util_in_bne = self.compute_utility(
            agent, obs_profile, bid_profile[idx]
        )

        metrics = ["l2_norm", "util_loss", "util_vs_bne", "util_in_bne"]
        values = [l2_norm, util_loss, util_vs_bne, util_in_bne]
        return metrics, values

    def get_bne(self, agent: str, obs: np.ndarray) -> np.ndarray:
        """Returns BNE for the respective setting

        Args:
            agent (str): agent
            obs (np.ndarray): observation

        Returns:
            np.ndarray: bne strategy given observation
        """
        return None

    def compute_l2_norm(self, agent: str, obs: np.ndarray, bids: np.ndarray) -> float:
        """compute approximated l2 norm of given bids compared to bne

        Args:
            agent (str): agent
            obs (np.ndarray): observations
            bids (np.ndarray): bids we want to compare to BNE

        Returns:
            float: approximated l2 norm
        """
        bne = self.get_bne(agent, obs)
        if bne is None:
            print("No BNE found in this setting")
            return np.nan
        else:
            return np.sqrt(1 / len(obs) * ((bids - bne) ** 2).sum())

    def compute_utility(
        self, agent: str, obs_profile: np.ndarray, bids: np.ndarray
    ) -> tuple:
        """compute metrics regarding utility

        Args:
            agent (str): agent
            obs_profile (np.ndarray): observation profile
            bids (np.ndarray): bids for agent

        Returns:
            Tuple[float, float, float]: relative utility loss, utility, utility in BNE
        """
        idx = self.bidder.index(agent)

        bid_profile = [
            self.get_bne(self.bidder[i], obs_profile[i]) for i in range(self.n_bidder)
        ]
        if any(bne is None for bne in bid_profile):
            print("No BNE found in this setting")
            return np.nan, np.nan, np.nan

        else:
            bid_profile = np.array(bid_profile)
            util_in_bne = self.utility(obs_profile[idx], bid_profile, idx).mean()

            bid_profile[idx] = bids
            util_vs_bne = self.utility(obs_profile[idx], bid_profile, idx).mean()

            if np.isclose(util_in_bne, 0.0):
                util_loss = np.nan
            else:
                util_loss = 1 - util_vs_bne / util_in_bne

            return util_loss, util_vs_bne, util_in_bne

    # ---------------------------------- methods for sampling of types ---------------------------------------- #

    def sample_types(self, n_vals: int) -> np.ndarray:
        """draw types, i.e. observations and/or valuations, for each agent according to the prior

        Args:
            n_vals (int): number of observations per agent

        Returns:
            np.ndarray: array that contains observations for each bidder (optional: common value with highest index)
        """

        if isinstance(self.prior, dict):
            raise NotImplementedError(
                "Different priors for different bidders are not yet implemented"
            )

        if self.prior == "uniform":
            return self.sample_types_uniform(n_vals)

        elif self.prior == "gaussian":
            return self.sample_types_gaussian(n_vals)

        elif self.prior == "gaussian_trunc":
            return self.sample_types_gaussian_trunc(n_vals)

        elif self.prior == "powerlaw":
            return self.sample_types_powerlaw(n_vals)

        elif self.prior == "affiliated_values":
            w = uniform.rvs(loc=0, scale=1, size=(3, n_vals))
            return np.array([w[0] + w[2], w[1] + w[2]])

        elif self.prior == "common_value":
            w = uniform.rvs(loc=0, scale=1, size=(self.n_bidder + 1, n_vals))
            return np.array([2 * w[i] * w[3] for i in range(self.n_bidder)] + [w[3]])

        else:
            raise ValueError('prior "' + self.prior + '" not implement')

    def sample_types_uniform(self, n_types: int) -> np.ndarray:
        """draw types according to uniform distribution
        Correlation (Bernoulli weights model) only implemented for 2 bidder

        Args:
            n_types (int): number of types per bidder

        Returns:
            np.ndarray: array with types for each bidder
        """
        if "corr" not in self.param_prior:
            return np.array(
                [
                    uniform.rvs(
                        loc=self.o_space[self.bidder[i]][0],
                        scale=self.o_space[self.bidder[i]][-1]
                        - self.o_space[self.bidder[i]][0],
                        size=n_types,
                    )
                    for i in range(self.n_bidder)
                ]
            )
        elif (
            ("corr" in self.param_prior)
            & (self.n_bidder == 2)
            & self.check_bidder_symmetric()
        ):
            gamma = self.param_prior["corr"]
            w = np.random.uniform(size=n_types)
            u = uniform.rvs(
                loc=self.o_space[self.bidder[0]][0],
                scale=self.o_space[self.bidder[0]][-1]
                - self.o_space[self.bidder[0]][0],
                size=(3, n_types),
            )
            return np.array(
                [
                    np.where(w < gamma, 1, 0) * u[2] + np.where(w < gamma, 0, 1) * u[0],
                    np.where(w < gamma, 1, 0) * u[2] + np.where(w < gamma, 0, 1) * u[1],
                ]
            )
        else:
            raise NotImplementedError(
                "Correlation with Uniform Prior only implemented for 2 bidder"
            )

    def sample_types_gaussian(self, n_types: int) -> np.ndarray:
        """draw types according to gaussian distribution

        Args:
            n_types (int): number of types per bidder

        Returns:
            np.ndarray: array with types for each bidder
        """
        if ("mu" in self.param_prior) & ("sigma" in self.param_prior):
            mu, sigma = self.param_prior["mu"], self.param_prior["sigma"]
        else:
            raise ValueError("'mu' and 'sigma' must be specified in param_prior")

        return np.array(
            [
                norm.rvs(
                    loc=mu,
                    scale=sigma,
                    size=n_types,
                )
                for i in range(self.n_bidder)
            ]
        )

    def sample_types_gaussian_trunc(self, n_types: int) -> np.ndarray:
        """draw types according to truncated gaussian distribution

        Args:
            n_types (int): number of types per bidder

        Returns:
            np.ndarray: array with types for each bidder
        """
        if ("mu" in self.param_prior) & ("sigma" in self.param_prior):
            mu, sigma = self.param_prior["mu"], self.param_prior["sigma"]
        else:
            raise ValueError("'mu' and 'sigma' must be specified in param_prior")

        return np.array(
            [
                truncnorm.rvs(
                    a=(self.o_space[i][0] - mu) / sigma,
                    b=(self.o_space[i][1] - mu) / sigma,
                    loc=mu,
                    scale=sigma,
                    size=n_types,
                )
                for i in self.bidder
            ]
        )

    def sample_types_powerlaw(self, n_types: int) -> np.ndarray:
        """draw types according to powerlaw distribution

        Args:
            n_types (int): number of types per bidder

        Returns:
            np.ndarray: array with types for each bidder
        """
        if "power" in self.param_prior:
            power = self.param_prior["power"]
        else:
            raise ValueError(
                "'power' must be specified in param_prior for powerlaw distribution"
            )

        return np.array(
            [
                powerlaw.rvs(
                    a=power,
                    loc=self.o_space[self.bidder[i]][0],
                    scale=self.o_space[self.bidder[i]][-1]
                    - self.o_space[self.bidder[i]][0],
                    size=n_types,
                )
                for i in range(self.n_bidder)
            ]
        )

    # --------------------------------------- helper methods ---------------------------------------------- #

    def check_bidder_symmetric(self, o_space=None) -> bool:
        """check if bidder have the same observation and action space

        Returns:
            bool:
        """
        agent_0 = self.bidder[0]
        o_space_identical = np.all(
            [self.o_space[agent_0] == self.o_space[i] for i in self.set_bidder]
        )
        a_space_identical = np.all(
            [self.a_space[agent_0] == self.a_space[i] for i in self.set_bidder]
        )

        if o_space:
            o_space_correct = self.o_space[agent_0] == o_space
            return o_space_identical and a_space_identical and o_space_correct
        else:
            return o_space_identical and a_space_identical

    def get_dimension_spaces(self) -> tuple:
        """
        Get dimension for observation and action space (dim_o, dim_a)
        """
        dim_o = (
            1
            if len(np.array(self.o_space[self.bidder[0]]).shape) == 1
            else np.array(self.o_space[self.bidder[0]]).shape[0]
        )
        dim_a = (
            1
            if len(np.array(self.a_space[self.bidder[0]]).shape) == 1
            else np.array(self.a_space[self.bidder[0]]).shape[0]
        )
        return dim_o, dim_a

    def get_bne(self, agent: str, obs: np.ndarray) -> np.ndarray:
        """return bne bids for given agent and observations

        Args:
            agent (str): bidder
            obs (np.ndarray): observations for bidder

        Returns:
            np.ndarray: equilibrium bids
        """
        return None

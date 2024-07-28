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
        self.value_model = "private"  # valuation depends only on private observation

    def __repr__(self) -> str:
        return f"Mechanism({self.name})"

    def __str__(self) -> str:
        general_info = f"Mechanism({self.name})\n- bidder: {self.bidder} \n"

        str_o_space = f"- observation space:\n"
        for key, value in self.o_space.items():
            str_o_space += f"   - {key}: {value}\n"

        str_a_space = f"- action space:\n"
        for key, value in self.a_space.items():
            str_a_space += f"   - {key}: {value}\n"

        str_param_prior = f"- prior\n"
        for key, value in self.param_prior.items():
            str_param_prior += f"   - {key}: {value}\n"

        str_param_util = f"- utility\n"
        for key, value in self.param_util.items():
            str_param_util += f"   - {key}: {value}\n"

        return (
            general_info + str_o_space + str_a_space + str_param_prior + str_param_util
        )

    # ------------------------------- methods for computation of utilities ------------------------------------- #

    def utility(
        self, obs_profile: np.ndarray, bids_profile: np.ndarray, index_agent: int
    ):
        """Compute utility for agent

        Args:
            obs_profile (np.ndarray): observations of all agents
            bids_profile (np.ndarray): bids of all agents
            index_agent (int): index of agent

        Returns:
            np.ndarry: utilities of agent (with index index_agent)
        """
        raise NotImplementedError

    def get_allocation(self, bids_profile: np.ndarray, index_agent: int) -> np.ndarray:
        """Compute alloction vector for agent

        Args:
            bids_profile (np.ndarray): bids of all agents
            index_agent (int): index of agent

        Returns:
            np.ndarray: allocation vector of agent (with index index_agent)
        """
        raise NotImplementedError

    def get_payment(
        self, bids_profile: np.ndarray, allocation_agent: np.ndarray, index_agent: int
    ) -> np.ndarray:
        """Compute payments for idx-th bidder

        Args:
            bids_profile (np.ndarray): bids of all agents
            allocation_agent (np.ndarray): allocation of agent
            index_agent (int): index of agent

        Returns:
            np.ndarray: payment vector of agent (with index index_agent)
        """
        raise NotImplementedError

    def test_input_utility(
        self, obs_profile: np.ndarray, bids_profile: np.ndarray, index_agent: int
    ):
        """Test input for utility function

        Args:
            obs_profile (np.ndarray): observations of all agents
            bids_profile (np.ndarray): bids of all agents
            index_agent (int): index of agent
        """
        if bids_profile.shape[0] != self.n_bidder:
            raise ValueError("wrong format of bids")
        elif index_agent >= self.n_bidder:
            raise ValueError("bidder with index " + str(index_agent) + " not avaible")
        pass

    def get_valuation(self, obs_profile: np.ndarray, index_agent: int) -> np.ndarray:
        """Determine valuations from observation profil.
        Depending on the types of interdependencies, we have different
        methods to get valuations.

        Args:
            obs_profile (np.ndarray): observation of all agents
            index_agent (int): index of agent

        Returns:
            np.ndarray: observations, possibly reformated
        """
        # Private values (valuations = observations)
        if self.value_model == "private":
            if len(obs_profile) != self.n_bidder:
                raise ValueError(
                    "obs_profile should contain only observations for all bidders"
                )
            return obs_profile[index_agent]

        # Common values, independent observations
        elif self.value_model == "common_independent":
            if len(obs_profile) != self.n_bidder + 1:
                raise ValueError(
                    "valuations should be included in obs_profile (last entry)"
                )
            valuations = obs_profile[-1]

        # Common values, correlated observations (valuations can be computed from observation_profiles)
        elif self.value_model == "common_affiliated":
            valuations = self.compute_valuations_from_observations(obs_profile)

        else:
            raise NotImplementedError(
                "get valuation only implemented for the private value model"
            )
        return valuations

    def compute_valuations_from_observations(
        self, obs_profile: np.ndarray
    ) -> np.ndarray:
        """For the common values with correlated observation case (see get_valuation()) we need a
        method to compute the valuations from the observation profile.

        Args:
            obs_profile (np.ndarray): _description_

        Returns:
            np.ndarray: common valuation
        """
        raise NotImplementedError

    # ---------------------------------- methods to compute metrics --------------------------------------- #

    def get_metrics_mechanism(
        self, obs_profile: np.ndarray, bid_profile: np.ndarray
    ) -> Dict[str, float]:
        """method to compute metrics regarding mechanism (e.g. revenue)"""
        return {}

    def get_metrics_agents(
        self, agent: str, obs_profile: np.ndarray, bid_profile: np.ndarray
    ) -> Dict[str, float]:
        """method to compute metrics regarding agents (e.g., utility, distance to BNE)"""
        return self.get_standard_metrics(agent, obs_profile, bid_profile)

    def get_standard_metrics(
        self, agent: str, obs_profile: np.ndarray, bid_profile: np.ndarray
    ) -> tuple:

        idx = self.bidder.index(agent)
        util, util_in_bne, util_vs_bne, util_loss = self.compute_utility_vs_bne(
            agent, obs_profile, bid_profile
        )
        l2_norm = self.compute_l2_norm(agent, obs_profile[idx], bid_profile[idx])

        metrics = [
            "utility",
            "utility_in_bne",
            "utility_vs_bne",
            "utility_loss_vs_bne",
            "l2_norm",
        ]
        values = [l2_norm, util_vs_bne, util_in_bne, util_loss, util]
        return dict(zip(metrics, values))

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
            return np.nan
        else:
            return np.sqrt(1 / len(obs) * ((bids - bne) ** 2).sum())

    def compute_utility_vs_bne(
        self, agent: str, obs_profile: np.ndarray, bid_profile: np.ndarray
    ) -> tuple:
        """compute metrics regarding utility

        Args:
            agent (str): agent
            obs_profile (np.ndarray): observation profile
            bid_profile (np.ndarray): bids for agent (other agents play according to BNE)

        Returns:
            Tuple[float, float, float]: relative utility loss, utility, utility in BNE
        """
        idx = self.bidder.index(agent)

        # all agents play computed strategy
        util = self.utility(obs_profile, bid_profile, idx).mean()

        bne_profile = [
            self.get_bne(self.bidder[i], obs_profile[i]) for i in range(self.n_bidder)
        ]
        if any(bne is None for bne in bid_profile):
            util_vs_bne, util_in_bne, util_loss = np.nan, np.nan, np.nan

        else:
            # all agents play bne
            bne_profile = np.array(bne_profile)
            util_in_bne = self.utility(obs_profile, bne_profile, idx).mean()

            # replace bne of agent idf with computed strategies
            bne_profile[idx] = bid_profile[idx]
            util_vs_bne = self.utility(obs_profile, bne_profile, idx).mean()

            if np.isclose(util_in_bne, 0.0):
                util_loss = np.nan
            else:
                util_loss = 1 - util_vs_bne / util_in_bne

        return util, util_in_bne, util_vs_bne, util_loss

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

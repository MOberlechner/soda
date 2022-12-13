import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from src.game import Game, discr_interval


class Strategy:
    """
    Strategies of the discretized game -> discrete distributional strategies
    Basically these are n x m matrices where each entry x[i,j] corresponds to the probability of playing action a[j] with valuation/observation o[i]

    - n,m (int) dimensions of the strategy (discretization parameter)
    - dim_o, dim_a (int) dimensions of the observation/action space
    - o_discr, a_discr (array) discretized observation/action space
    - prior (array) vector with marginal prior
    - x (array) current strategy
    - utility, utility loss (list)
    - history (list) all intermediate strategies

    """

    def __init__(self, agent: str, game: Game):
        """Create strategy for agent.
        Parameters are given by respective game.

        Args:
            agent (str): name of repr. bidder/agent
            game (Game): approximation game
        """

        # name of the bidder
        self.agent = agent

        # observation and action space
        self.o_discr = game.o_discr[agent]
        self.a_discr = game.a_discr[agent]

        # number of discretization points
        self.n = game.n
        self.m = game.m

        # dimension of spaces
        self.dim_o = game.dim_o
        self.dim_a = game.dim_a

        # prior (marginal) distribution
        self.prior = game.prior[agent]

        # strategy - primal iterate
        self.x = -np.ones(tuple([game.n] * self.dim_o + [game.m] * self.dim_a)) / (
            game.n**self.dim_o * game.m * self.dim_a
        )
        # strategy - dual iterate
        self.y = np.zeros_like(self.x)

        # utility, history, gradients
        (
            self.utility,
            self.utility_loss,
            self.dist_prev_iter,
            self.history,
            self.history_gradient,
        ) = ([], [], [], [], [])

    def __repr__(self) -> str:
        return f"Strategy({self.agent})"

    def __str__(self):
        return "Strategy Bidder " + self.agent + " - shape: " + str(self.x.shape)

    def margin(self):
        """
        Get marginal distribution over observations
        """
        return self.x.sum(axis=tuple(range(self.dim_o, self.dim_o + self.dim_a)))

    def initialize(
        self, init_method: str, param: dict = {}, lower_bound: float = 1e-20
    ):
        """
        Initialize strategy according to different methods. The sum of probabilities over all actions given a valuation is determined by the prior.
        But the probability P(bid|val) can be set arbitrarily:
        - equal: for a given valuation, all actions are played with equal probability.
        - random: for a given valuation, all actions are played with a random probability p ~ U([0,1]), which are scaled accordingly
        - function: discrete distr. strategy that corresponds approximately to function (given in param)
        - matrix: discrete distr. strategy given in param

        Parameters
        ----------
        init_method : strm specifies method such as equal, random, trufhul or function
        param : dict, used for method function, contains evaluation of function
        lower_bound : float,  used for method function

        Returns
        -------

        """
        # new initializations deletes histories of strategies, utilities, etc
        (
            self.utility,
            self.utility_loss,
            self.dist_prev_iter,
            self.history,
            self.history_dual,
            self.history_gradient,
            self.history_best_response,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        if init_method == "random":
            sigma = np.random.uniform(0, 1, size=self.x.shape)

        elif init_method == "random_no_overbid":
            if self.dim_o == self.dim_a == 1:
                aa, oo = np.meshgrid(self.a_discr, self.o_discr)
                sigma = np.random.uniform(0, 1, size=self.x.shape)
                sigma[np.array(oo <= aa)] = lower_bound
            else:
                raise NotImplementedError(
                    "random_no_overbid only available for 1-dim action and observation space"
                )

        elif init_method == "equal":
            sigma = np.ones(self.x.shape)

        elif init_method == "truthful":
            if self.dim_o == self.dim_a == 1:
                n, m = self.x.shape
                sigma = np.array(
                    [
                        [
                            np.exp(-1 / 2 * ((i - j) / 0.02) ** 2)
                            for j in np.linspace(0, 1, m)
                        ]
                        for i in np.linspace(0, 1, n)
                    ]
                )
            else:
                raise NotImplementedError("truthful only implement for 1-dim case")

        elif init_method == "truthful_inv":
            if self.dim_o == self.dim_a == 1:
                n, m = self.x.shape
                sigma = np.array(
                    [
                        [
                            np.exp(-1 / 2 * ((i - j) / 0.02) ** 2)
                            for j in np.linspace(0, 1, m)
                        ]
                        for i in np.linspace(1, 0, n)
                    ]
                )

        elif init_method == "function":
            b = param["init_function"]
            n, m = self.x.shape
            sigma = lower_bound * np.ones((n, m))
            for i in range(n):
                idx = (np.abs(self.a_discr - b[i])).argmin()
                sigma[i, idx] = 1

        elif init_method == "matrix":
            if (
                (param["init_matrix"].shape[0] == self.n)
                & (param["init_matrix"].shape[-1] == self.m)
                & (
                    param["init_matrix"].size
                    == self.n**self.dim_o * self.m**self.dim_a
                )
            ):
                sigma = param["init_matrix"]
            else:
                raise ValueError("Dimension of matrix is not as expected")

        else:
            raise ValueError("init_method not known")

        # normalize strategy according to prior
        sigma_sum = sigma.sum(axis=tuple(range(self.dim_o, self.dim_o + self.dim_a)))
        self.x = (1 / sigma_sum * self.prior).reshape(
            list(self.prior.shape) + [1] * self.dim_a
        ) * sigma

        # init dual variable
        self.y = np.zeros_like(self.x)

    # --------------------------------------- METHODS USED FOR COMPUTATION ------------------------------------------- #

    def best_response(self, gradient: np.ndarray) -> np.ndarray:
        """Compute Best response given the gradient.
        The best response given the opponents strategies is the solution of a LP.
        Due to the simple structure it can be computed directly:
        pick for each valuation the action with the highest expected utility.

        Args:
            gradient (np.ndarray): gradient given current strategy profile

        Returns:
            np.ndarray: best response
        """
        # create array for best response
        best_response = np.zeros(gradient.shape)

        # determine largest entry of gradient for each valuation und put all the weight on the respective entry
        if self.dim_o == self.dim_a == 1:
            index_max = gradient.argmax(axis=1)
            best_response[range(self.n), index_max] = self.prior

        elif self.dim_o == 1:
            for i in range(self.n):
                index_max = np.unravel_index(
                    np.argmax(gradient[i], axis=None), gradient[i].shape
                )
                best_response[i][index_max] = self.margin()[i]

        else:
            raise NotImplementedError(
                "Best Response not implemented for multi-dimensional observation space"
            )

        return best_response

    def empirical_mean(self, iter: int = 1) -> np.ndarray:
        """Compute empirical mean of strategies
        over a number of iterations
        iter = 1 : only last iterate (= self.x)
        iter = -1: mean over all iterates

        Args:
            iter (int): number of iterations to consider

        Returns:
            np.ndarray: empirical mean of strategy
        """
        # return last iterate
        if iter == 1 or (len(self.history) == 0):
            return self.x
        # return mean over all iterates
        elif iter == -1:
            return np.mean(self.history, axis=0)
        elif iter > 1:
            return np.mean(self.history[-iter:], axis=0)
        else:
            raise ValueError

    # --------------------------------------- METHODS USED TO DURING ITERATIONS ---------------------------------------- #
    def update_utility(self, gradient: np.ndarray):
        """
        Compute utility for current strategy and add to list self.utility
        """
        self.utility += [(self.x * gradient).sum()]

    def update_utility_loss(self, gradient: np.ndarray):
        """
        Compute relative utility loss for current strategy and add to list self.utility
        Add 1e-50 so that we don't divide by zero
        """
        util_br = (self.best_response(gradient) * gradient).sum()
        self.utility_loss += [
            np.abs(
                1
                - (self.x * gradient).sum()
                / (util_br if not np.isclose(util_br, 0, atol=1e-20) else 1e-20)
            )
        ]

    def update_dist_prev_iter(self):
        """
        Compute Euclidean distance to previous iterate
        We assume that update_history is performed after this computation
        """
        self.dist_prev_iter += [
            np.linalg.norm(self.x - self.history[-1])
            if len(self.history) > 0
            else np.nan
        ]

    def update_history_strategy(self, update_history_bool: bool):
        """
        Add current strategy to history of primal iterates

        Args:
            update_history_bool: If true we save all history,
                else we save only initial strategy and last iterate
        """
        if update_history_bool or (len(self.history) < 2):
            self.history += [self.x]
        else:
            self.history[1] = self.x

    def update_history_dual(self, update_history_bool: bool):
        """
        Add current dual iterate to history of dual iterates

        Args:
            update_history_bool: If true we save all history,
                else we save only initial strategy and last iterate
        """
        if update_history_bool or (len(self.history_dual) < 2):
            self.history_dual += [self.y]
        else:
            self.history_dual[1] = self.y

    def update_history_gradient(self, gradient: np.ndarray, update_history_bool: bool):
        """
        Add current gradient to history of gradients
        """
        if update_history_bool or (len(self.history_gradient) < 2):
            self.history_gradient += [gradient]
        else:
            self.history_gradient[1] = gradient

    def update_history(self, gradient: np.ndarray, update_history_bool: bool):
        """
        Call all update methods

        Args:
            gradient: gradient to compute util, ...
            update_history_bool: if True all history are saved, else only util (loss)
        """
        self.update_utility(gradient)
        self.update_utility_loss(gradient)
        self.update_dist_prev_iter()
        self.update_history_strategy(update_history_bool)
        self.update_history_dual(update_history_bool)
        self.update_history_gradient(gradient, update_history_bool)

    # --------------------------------------- METHODS USED TO ANALYZE RESULTS ---------------------------------------- #

    def bid(self, observation: np.ndarray):
        """
        Sample bids from the strategy

        Parameters
        ----------
        observation : np.ndarray, observations

        Returns
        -------
        np.ndarray, returns bids sampled from the respective mixed strategies given through the observations
        """

        # number of discretization points observations
        n = self.x.shape[0]

        if self.dim_o == 1:
            # determine corresponding discrete observation
            idx_obs = np.floor(
                (observation - self.o_discr[0])
                / (self.o_discr[-1] - self.o_discr[0])
                * n
            ).astype(int)
            idx_obs = np.maximum(0, np.minimum(n - 1, idx_obs))

            uniques, counts = np.unique(
                idx_obs, return_inverse=False, return_counts=True
            )

            if self.dim_a == 1:

                bids = np.zeros(idx_obs.shape)
                for d, c in zip(uniques, counts):
                    bids[idx_obs == d] = np.random.choice(
                        self.a_discr, size=c, p=self.x[d] / self.x[d].sum()
                    )
                return bids

            elif self.dim_a == 2:

                bids = np.zeros((len(idx_obs), 2))
                a_discr_2d = np.array(
                    [(a2, a1) for a1 in self.a_discr[0] for a2 in self.a_discr[1]]
                )

                for d, c in zip(uniques, counts):
                    idx_act = np.random.choice(
                        range(self.m**2),
                        size=c,
                        p=self.x[d].reshape(-1) / self.x[d].sum(),
                    )
                    # we have to switch order back (first single, second split)
                    bids[idx_obs == d] = np.array(a_discr_2d[idx_act])[0][[1, 0]]

                return bids.T

            else:
                raise NotImplementedError(
                    "Bids can only be sampled for one- or two-dimensional actions"
                )

        else:
            raise NotImplementedError(
                "Bids can only be sampled for one-dimensional observations"
            )

    def plot(
        self,
        metrics: bool = False,
        grad: bool = False,
        beta: np.ndarray = None,
        iter: int = None,
        save: bool = False,
    ):
        """Visualize Strategy

        Args:
            metrics (bool, optional): show metrics (util_loss, dist_prev_iter). Defaults to False.
            grad (bool, optional): show gradient with best response. Defaults to False.
            beta (np.ndarray, optional): plot function over strategy. Defaults to None.
            iter (int, optional): show intermediate strategy. Defaults to None.
            save (bool, optional): save plot. Defaults to False.

        Raises:
            NotImplementedError: plot not available for multi-dim strategies
            ValueError: history not available
        """

        param = {
            "fontsize_title": 14,
            "fontsize_legend": 13,
            "fontsize_label": 12,
        }

        # check input
        if grad and (self.dim_a + self.dim_o != 2):
            raise NotImplementedError(
                "Plot of Gradient only for 1-dim actions and observations available"
            )

        # choose correct strategy and gradient from history or take current one
        if iter is None:
            strategy = self.x
            if grad:
                gradient = self.history_gradient[-1]
        elif iter == 0:
            strategy = self.history[0]
            if grad:
                gradient = self.history_gradient[0]
        else:
            if len(self.history) != len(self.utility):
                raise ValueError(
                    "History not saved. Intermediate strategies are not available"
                )
            else:
                if iter == -1:
                    strategy = self.empirical_mean()
                    if grad:
                        gradient = np.array(self.history_gradient).mean(axis=0)
                else:
                    strategy = self.history[iter]
                if grad:
                    gradient = self.history_gradient[iter]

        # create figure
        num_plots = self.dim_a + metrics + grad
        counter = 1
        fig = plt.figure(figsize=(5 * num_plots, 5), tight_layout=True)

        # plot strategy
        if self.dim_a == 1:
            ax_strat = fig.add_subplot(1, num_plots, counter)
            self._plot_strategy(ax_strat, strategy, param, iter, beta)
            counter += 1
        elif self.dim_a == 2:
            # Special case: Split Award Auction
            ax_strat = fig.add_subplot(1, num_plots, counter)
            self._plot_strategy(
                ax_strat, strategy.sum(axis=2), param, iter, beta, axis_a=0
            )
            counter += 1
            ax_strat = fig.add_subplot(1, num_plots, counter)
            self._plot_strategy(
                ax_strat, strategy.sum(axis=1), param, iter, beta, axis_a=1
            )
            counter += 1

        else:
            raise NotImplementedError("Visualization not implemented for dim_a > 2")

        # plot gradient
        if grad:
            ax_grad = fig.add_subplot(1, num_plots, counter)
            self._plot_gradient(ax_grad, gradient, param, iter)
            counter += 1

        # plot metrics
        if metrics:
            ax_metr = fig.add_subplot(1, num_plots, counter)
            self._plot_metrics(ax_metr, param, iter)

        # save figure
        if save:
            plt.savefig(
                f"plot_strategy_{self.agent}.png",
                facecolor="white",
                transparent=False,
                dpi=150,
            )

    def _plot_strategy(
        self,
        ax,
        strategy: np.ndarray,
        param: dict,
        iter: int = None,
        beta=None,
        axis_a: int = 0,
        axis_o: int = 0,
    ) -> None:
        """Plot Gradient (for standard incomplete information setting)

        Args:
            ax: axis from pyplot
            strategy (np.ndarray): Strategy to visualize
            param (dict): specifies parameter (fontsize, ...) for plots
            iter (int, optional): show intermediate strategy. Defaults to None.
            beta (function, optional): Defaults to None.
            axis_a (int, optional): Which axis to visualize, if multidimensional action space Defaults to 1.
        """
        # plot strategy
        if self.dim_o > 1:
            raise NotImplementedError(
                "Visualization only implemented for dim_a = 1,2 and dim_o = 1"
            )

        if self.dim_a == 1:
            param_extent = (
                self.o_discr[0],
                self.o_discr[-1],
                self.a_discr[0],
                self.a_discr[-1],
            )
        else:
            param_extent = (
                self.o_discr[0],
                self.o_discr[-1],
                self.a_discr[axis_a][0],
                self.a_discr[axis_a][-1],
            )
        ax.imshow(
            strategy.T / self.margin(),
            extent=param_extent,
            origin="lower",
            vmin=0,
            cmap="Greys",
            aspect="auto",
        )

        # plot function
        if beta is not None:
            if callable(beta):
                b = beta(self.o_discr)
            else:
                b = beta
            if self.dim_a == 2:
                b = b[axis_a]
            ax.plot(
                self.o_discr,
                b,
                linestyle="--",
                color="tab:orange",
                linewidth=2,
                label="analyt. BNE",
            )
            ax.legend(fontsize=param["fontsize_legend"])

        # labels
        title_label = f"Strategy - Agent {self.agent} " + (
            f"(Iteration {iter})" if iter is not None else f""
        )
        ax.set_title(
            title_label, fontsize=param["fontsize_title"], verticalalignment="bottom"
        )
        ax.set_xlabel("Observation o", fontsize=param["fontsize_label"])
        ax.set_ylabel("Bid b", fontsize=param["fontsize_label"])

    def _plot_gradient(
        self, ax, gradient: np.ndarray, param: dict, iter: int = None
    ) -> None:
        """Plot Gradient (for standard incomplete information setting)

        Args:
            ax: axis from pyplot
            gradient (np.ndarray): gradient
            param (dict): specifies parameter (fontsize, ...) for plots
            iter (int, optional): show intermediate gradient. Defaults to None.
        """

        # plot gradient
        im = ax.imshow(
            gradient.T,
            extent=(
                self.o_discr[0],
                self.o_discr[-1],
                self.a_discr[0],
                self.a_discr[-1],
            ),
            cmap="RdBu",
            origin="lower",
            aspect="auto",
            vmin=-np.abs(gradient).max(),
            vmax=+np.abs(gradient).max(),
        )
        # plot best response
        best_response = self.best_response(gradient)
        best_response[best_response == 0] = np.nan
        ax.imshow(
            best_response.T / self.margin(),
            cmap="Wistia",
            vmin=0,
            vmax=1.1,
            extent=(
                self.o_discr[0],
                self.o_discr[-1],
                self.a_discr[0],
                self.a_discr[-1],
            ),
            origin="lower",
            aspect="auto",
        )
        ax.plot(
            [], [], color="tab:orange", linewidth=0, marker="s", label="best\nresponse"
        )

        # labels
        title_label = f"Gradient - Agent {self.agent} " + (
            f"(Iteration {iter})" if iter is not None else f""
        )
        ax.set_title(
            title_label, fontsize=param["fontsize_title"], verticalalignment="bottom"
        )
        ax.set_xlabel("Observation o", fontsize=param["fontsize_label"])
        ax.set_ylabel("Bid b", fontsize=param["fontsize_label"])

        # grid, legend, ticks
        ax.legend(loc="upper left", fontsize=param["fontsize_legend"])

    def _plot_metrics(self, ax, param: dict, iter: int = None) -> None:
        """Plot Metrics

        Args:
            ax (_type_): _description_
            param (dict): _description_
            iter (int, optional): _description_. Defaults to None.
        """
        # plot metrics
        ax.plot(
            self.utility_loss,
            linestyle="-",
            linewidth=1.5,
            color="k",
            label="util. loss",
        )
        ax.plot(
            self.dist_prev_iter,
            linestyle="--",
            linewidth=1.5,
            color="k",
            label="dist. prev. iter.",
        )
        if iter is not None:
            ax.axvline(x=iter, color="tab:orange", linestyle=":", linewidth=2)

        # labels
        title_label = f"Metrics for Equilibrium Learning"
        ax.set_title(
            title_label, fontsize=param["fontsize_title"], verticalalignment="bottom"
        )
        ax.set_xlabel("Iterations", fontsize=param["fontsize_label"])

        # grid, legend, ticks
        ax.set_yscale("log")
        ax.set_ylim(top=1)
        ax.grid(axis="y", linestyle="-", linewidth=0.5, color=".25", zorder=-10)
        ax.legend(loc="upper right", fontsize=param["fontsize_legend"])

    def save(self, name: str, setting: str, path: str, save_init: bool = False):
        """Saves strategy in respective directory

        Parameters
        ----------
        name : str, name of strategy
        path : str ,path to directory (where directory strategies is contained)
        setting : str, subdirectory in strategies
        save_init: bool, save initial strategy as well
        """
        np.save(
            path
            + "strategies/"
            + setting
            + "/"
            + name
            + "_agent_"
            + self.agent
            + ".npy",
            self.x,
        )
        if save_init:
            np.save(
                path
                + "strategies/"
                + setting
                + "/"
                + name
                + "_agent_"
                + self.agent
                + "_init.npy",
                self.history[0],
            )

    def load(self, name: str, setting: str, path: str):
        """Load strategy from respective directory, same naming convention as in save method

        Parameters
        ----------
        name : str, name of strategy
        setting : str, subdirectory of strategies, mechanism
        path : str, path to directory (in which strategies/ is contained

        Returns
        -------

        """
        # current directory:
        current_dir = os.getcwd()
        # path to project
        dir_project = current_dir.split("soda")[0] + "soda/"

        try:
            self.x = np.load(
                dir_project
                + path
                + "strategies/"
                + setting
                + "/"
                + name
                + "_agent_"
                + self.agent
                + ".npy"
            )
        except:
            print(
                'File: "'
                + name
                + "_agent_"
                + self.agent
                + ".npy"
                + '" is not available in directory "'
                + dir_project
                + path
                + "strategies/"
                + setting
                + "/"
                + '"'
            )

    def load_scale(self, name: str, setting: str, path: str, n_scaled: int, m_scaled):
        """Load strategy from respective directory, same naming convention as in save method
        Should be used if saved strategy has a lower discretization than the strategy we have

        Parameters
        ----------
        name : str, name of strategy
        setting: str, name of setting (subdirectory in strategies)
        path : str, path to directory (in which strategies/ is contained
        n_scaled: int, discretization (type) in the larger setting
        m_scaled: int, discretization (action) in the larger setting

        Returns
        -------

        """
        try:
            strat = np.load(
                path
                + "strategies/"
                + setting
                + "/"
                + name
                + "_agent_"
                + self.agent
                + ".npy"
            )
            bool_strat_loaded = True

        except:
            bool_strat_loaded = False
            print(
                'File: "'
                + name
                + "_agent_"
                + self.agent
                + ".npy"
                + '" is not available in directory "'
                + path
                + "strategies/"
                + '"'
            )

        # only for two dimensional strategies
        if bool_strat_loaded:
            if len(self.x.shape) == 2:
                # idea: put probability measure of action two closest action in new discretization
                n, m = strat.shape
                a_discr = discr_interval(
                    self.a_discr[0], self.a_discr[-1], m, midpoint=False
                )
                strat_new = np.zeros((n_scaled, m_scaled))
                factor_n = int(n_scaled / n)

                if factor_n * n != n_scaled:
                    raise ValueError(
                        "error in load_scale: n_scaled is not a multiple of n"
                    )

                for j in range(m):
                    j_star = int(np.argmin(np.abs(a_discr[j] - self.a_discr)))
                    strat_new[:, j_star] = np.repeat(strat[:, j], factor_n)

                strat_new_sum = strat_new.sum(axis=1)
                self.x = (1 / strat_new_sum * self.prior).reshape(
                    (n_scaled, 1)
                ) * strat_new

            else:
                raise NotImplementedError(
                    "Scaling is only implemented for 1-dim type and action space!"
                )

import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from soda.game import Game


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
        self.agent = agent
        self.game = game
        self.mechanism = game.mechanism

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

        # current gradient
        self.gradient = np.nan * np.ones_like(self.x)

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
        # overwrite old history
        self.prepare_history(max_iter=2, save_history_bool=True)

        if init_method == "random":
            sigma = np.random.uniform(0, 1, size=self.x.shape)

        elif init_method == "nan":
            sigma = np.nan * np.ones(self.x.shape)

        elif init_method == "random_no_overbid":
            if self.dim_o == self.dim_a == 1:
                aa, oo = np.meshgrid(self.a_discr, self.o_discr)
                sigma = np.random.uniform(0, 1, size=self.x.shape)
                sigma[np.array(oo < aa)] = lower_bound
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
            if self.dim_o == self.dim_a == 1:
                sigma = lower_bound * np.ones_like(self.x)
                for i in range(self.n):
                    idx = (np.abs(self.a_discr - b[i])).argmin()
                    sigma[i, idx] = 1

            elif (self.dim_o == 1) & (self.dim_a == 2):
                sigma = lower_bound * np.ones_like(self.x)
                for i in range(self.n):
                    idx1 = (np.abs(self.a_discr[0] - b[0][i])).argmin()
                    idx2 = (np.abs(self.a_discr[1] - b[1][i])).argmin()
                    sigma[i, idx1, idx2] = 1

            else:
                raise NotImplementedError(
                    "init_function only for 1-dim observation and action spaces available"
                )

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
        best_response = np.zeros_like(gradient)

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
        elif self.save_history_bool:
            if iter == -1:
                return np.nanmean(self.history, axis=0)
            elif iter > 1:
                return np.nanmean(self.history[-iter:], axis=0)
            else:
                raise ValueError
        else:
            raise ValueError(
                "Empirical mean only available if save_history_bool is True"
            )

    # --------------------------------------- METHODS USED TO UPDATE METRICS ---------------------------------------- #
    def get_utility(self):
        """Compute current utility"""
        return (self.x * self.gradient).sum()

    def get_utility_loss(self):
        """Compute and save relative utility loss for current strategy
        Add 1e-20 so that we don't divide by zero
        """
        util_br = (self.best_response(self.gradient) * self.gradient).sum()
        util = (self.x * self.gradient).sum()
        return np.abs(
            1 - util / (util_br if not np.isclose(util_br, 0, atol=1e-20) else 1e-20)
        )

    def update_dist_prev_iter(self, t: iter):
        """Compute and save Euclidean distance to previous iteration
        If history of strategies is not save (i.d., save_history_bool is False),
        then we only store the the initial and the last strategy in self.history

        Args:
            t (iter): current iteration
        """
        if t > 0:
            if self.save_history_bool:
                self.dist_prev_iter[t] = np.linalg.norm(self.x - self.history[t - 1])
            else:
                self.dist_prev_iter[t] = np.linalg.norm(self.x - self.history[1])

    def update_history_strategy(self, t: int):
        """Save current stratety
        If save_history_bool if False, we only story the initial and the last strategy
        in a list of length 2

        Args:
            t (int): current iteration
        """
        t = t if self.save_history_bool else min(t, 1)
        self.history[t] = self.x

    def update_history_dual(self, t: int):
        """
        Add current dual iterate to history of dual iterates

        Args:
            t (int): currnet iteration
        """
        t = t if self.save_history_bool else min(t, 1)
        self.history_dual[t] = self.y

    def update_history_gradient(self, t: int):
        """Save current gradient

        Args:
            t (int): currnet iteration
            gradient (np.ndarray): current gradient
        """
        t = t if self.save_history_bool else min(t, 1)
        self.history_gradient[t] = self.gradient

    def update_history(self, t: int):
        """Update all histories

        Args:
            t (int): iteration
            gradient (np.ndarray): current gradient
            save_history_bool (bool): save history of strategies, gradients, ...
        """
        self.utility[t] = self.get_utility()
        self.utility_loss[t] = self.get_utility_loss()

        self.update_dist_prev_iter(
            t,
        )
        self.update_history_strategy(
            t,
        )
        self.update_history_dual(t)
        self.update_history_gradient(t)

    def prepare_history(self, max_iter: int, save_history_bool: bool) -> None:
        """Create arrays to store history.
        Allocation the memory at the beginning should make this faster

        Args:
            save_history_bool (bool): save history of gradients/strategies as well
        """
        self.save_history_bool = save_history_bool
        (self.utility, self.utility_loss, self.dist_prev_iter,) = (
            np.nan * np.ones(max_iter),
            np.nan * np.ones(max_iter),
            np.nan * np.ones(max_iter),
        )
        iterations = max_iter if save_history_bool else 2

        (
            self.history,
            self.history_dual,
            self.history_gradient,
            self.history_best_response,
        ) = (
            np.nan
            * np.ones(
                tuple([iterations] + [self.n] * self.dim_o + [self.m] * self.dim_a)
            ),
            np.nan
            * np.ones(
                tuple([iterations] + [self.n] * self.dim_o + [self.m] * self.dim_a)
            ),
            np.nan
            * np.ones(
                tuple([iterations] + [self.n] * self.dim_o + [self.m] * self.dim_a)
            ),
            np.nan
            * np.ones(
                tuple([iterations] + [self.n] * self.dim_o + [self.m] * self.dim_a)
            ),
        )

    # --------------------------------------- METHODS USED FOR SIMULATIONS ---------------------------------------- #

    def sample_bids(self, observation: np.ndarray):
        """
        Sample bids from the strategy

        Parameters
        ----------
        observation : np.ndarray, observations

        Returns
        -------
        np.ndarray, returns bids sampled from the respective mixed strategies given through the observations
        """
        if self.dim_o == 1:

            idx_obs = self._find_nearest_discrete_point(observation, self.o_discr)
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

    def _find_nearest_discrete_point(
        self, vec_continuous: np.ndarray, vec_discrete: np.ndarray
    ) -> np.ndarray:
        """

        Args:
            vec_continuous (np.ndarray): continuous values
            vec_discrete (np.ndarray): discrete values

        Returns:
            np.ndarray: index of discrete values
        """
        idx_obs = np.floor(
            (vec_continuous - vec_discrete.min())
            / (vec_discrete.max() - vec_discrete.min())
            * len(vec_discrete)
        ).astype(int)
        return np.maximum(0, np.minimum(len(vec_discrete) - 1, idx_obs))

    # -------------------------------------- METHODS USED TO VISUALIZE STRATEGIES --------------------------------------- #

    def plot(
        self,
        metrics: bool = False,
        grad: bool = False,
        beta: np.ndarray = None,
        iter: int = None,
        save: bool = False,
        save_path: str = "strategy",
    ):
        """Visualize Strategy

        Args:
            metrics (bool, optional): show metrics (util_loss, dist_prev_iter). Defaults to False.
            grad (bool, optional): show gradient with best response. Defaults to False.
            beta (np.ndarray, optional): plot function over strategy. Defaults to None.
            iter (int, optional): show intermediate strategy. Defaults to None.
            save (bool, optional): save plot. Defaults to False.
            save_path (bool, optional): path to save_plot. Defaults to

        Raises:
            NotImplementedError: plot not available for multi-dim strategies
            ValueError: history not available
        """

        param = {
            "fontsize_title": 14,
            "fontsize_legend": 13,
            "fontsize_label": 12,
        }

        # choose correct strategy and gradient from history or take current one
        strategy, gradient = self._get_elements_from_history(iter, grad)

        # create figure
        num_plots = self.dim_a + metrics + grad * self.dim_a
        counter = 1
        fig = plt.figure(figsize=(5 * num_plots, 5), tight_layout=True)

        # plot strategy with 1-dim action space
        if self.dim_a == 1:
            ax_strat = fig.add_subplot(1, num_plots, counter)
            if self.n > 1:
                self._plot_strategy(ax_strat, strategy, param, iter, beta)
            else:
                self._plot_strategy_complete_info(ax_strat, strategy, param, iter, beta)
            counter += 1
        # plot strategy with 2-dim action space
        elif self.dim_a == 2:
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

        # plot gradient with 1-dim action space
        if grad & (self.dim_a == 1):
            ax_grad = fig.add_subplot(1, num_plots, counter)
            if self.n > 1:
                self._plot_gradient(ax_grad, gradient, param, iter)
            else:
                self._plot_gradient_complete_info(ax_grad, gradient, param, iter)
            counter += 1
        elif grad & (self.dim_a == 2):
            ax_grad = fig.add_subplot(1, num_plots, counter)
            self._plot_gradient(ax_grad, gradient, param, iter, axis_a=0)
            counter += 1
            ax_grad = fig.add_subplot(1, num_plots, counter)
            self._plot_gradient(ax_grad, gradient, param, iter, axis_a=1)
            counter += 1

        # plot metrics
        if metrics:
            ax_metr = fig.add_subplot(1, num_plots, counter)
            self._plot_metrics(ax_metr, param, iter)

        # save figure
        if save:
            plt.savefig(
                f"{save_path}.png",
                facecolor="white",
                transparent=False,
                dpi=75,
            )
            plt.close(fig)

    def _get_elements_from_history(self, iter: int, grad: bool) -> tuple:
        """Get Strategy and Gradient (opt.) from history to plot

        Args:
            iter (int): _description_
            bool (grad): _description_

        Returns:
            Tuple: _description_
        """
        if (len(self.utility) > len(self.history)) & (iter not in [0, None]):
            raise ValueError(
                f"Cannot plot strategy/gradient for iteration {iter} since history was not saved"
            )

        elif iter is None:
            strategy = self.x
            gradient = self.gradient

        elif iter == -1:
            strategy = self.empirical_mean(iter=-1)
            gradient = gradient = np.nanmean(self.history_gradient, axis=0)

        elif iter >= 0:
            strategy = self.history[iter]
            gradient = self.history_gradient[iter]

        else:
            raise ValueError(f"Iteration {iter} not feasible to plot")

        return strategy, gradient

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
            axis_a (int, optional): Which axis to visualize, if multidimensional action space Defaults to 0.
        """
        # plot strategy
        if self.dim_o > 1:
            raise NotImplementedError(
                "Visualization only implemented for dim_a = 1,2 and dim_o = 1"
            )
        param_extent = self._get_imshow_extent(axis_a)
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
            ax.set_ylim(param_extent[2], param_extent[3])

        # labels
        title_label = (
            f"Strategy - Agent {self.agent}"
            + (f", Bid {axis_a+1}" if self.dim_a > 1 else f"")
            + (
                ""
                if iter is None
                else (f" (Emp. Mean)" if iter == -1 else f" (Iteration {iter})")
            )
        )
        ax.set_title(
            title_label, fontsize=param["fontsize_title"], verticalalignment="bottom"
        )
        ax.set_xlabel("Observation o", fontsize=param["fontsize_label"])
        ax.set_ylabel("Bid b", fontsize=param["fontsize_label"])

    def _plot_strategy_complete_info(
        self,
        ax,
        strategy: np.ndarray,
        param: dict,
        iter: int = None,
        beta=None,
    ) -> None:
        """Plot Gradient (for standard incomplete information setting)

        Args:
            ax: axis from pyplot
            strategy (np.ndarray): Strategy to visualize
            param (dict): specifies parameter (fontsize, ...) for plots
            iter (int, optional): show intermediate strategy. Defaults to None.
            beta (function, optional): Defaults to None.
            axis_a (int, optional): Which axis to visualize, if multidimensional action space Defaults to 0.
        """
        # plot strategy
        if self.dim_a > 1:
            raise NotImplementedError(
                "Visualization for complete information (n=1) case only implemented for dim_a = 1"
            )

        ax.bar(
            self.a_discr,
            strategy[0],
            color="gray",
            width=(self.a_discr[-1] - self.a_discr[0]) / self.m,
        )

        # plot function
        if beta is not None:
            ax.axvline(
                x=beta,
                linestyle="--",
                color="tab:orange",
                linewidth=4,
                label="analyt. BNE",
            )
            ax.legend(loc="upper left", fontsize=param["fontsize_legend"])

        # labels
        title_label = (
            f"Strategy - Agent {self.agent}"
            + f" (o={self.o_discr[0]:.2f})"
            + (
                ""
                if iter is None
                else (f" (Emp. Mean)" if iter == -1 else f" (Iteration {iter})")
            )
        )
        ax.set_title(
            title_label, fontsize=param["fontsize_title"], verticalalignment="bottom"
        )
        ax.set_xlabel("Bid b", fontsize=param["fontsize_label"])
        ax.set_ylabel("Probability", fontsize=param["fontsize_label"])

    def _plot_gradient(
        self,
        ax,
        gradient: np.ndarray,
        param: dict,
        iter: int = None,
        axis_a: int = 0,
    ) -> None:
        """Plot Gradient (for standard incomplete information setting)

        Args:
            ax: axis from pyplot
            gradient (np.ndarray): gradient
            param (dict): specifies parameter (fontsize, ...) for plots
            iter (int, optional): show intermediate gradient. Defaults to None.
            axis_a (int, optional): Which axis to visualize, if multidimensional action space Defaults to 0.
        """
        gradient_plot = gradient if self.dim_a == 1 else gradient.sum(axis=2 - axis_a)
        param_extent = self._get_imshow_extent(axis_a)

        # plot gradient
        ax.imshow(
            gradient_plot.T,
            extent=param_extent,
            cmap="RdBu",
            origin="lower",
            aspect="auto",
            vmin=-np.abs(gradient_plot).max(),
            vmax=+np.abs(gradient_plot).max(),
        )
        # plot best response
        best_response = self.best_response(gradient)

        # best_respone: matrix, best_response_y: vector
        if self.dim_a == 1:
            best_response_y = [
                self.a_discr[a_idx] for a_idx in best_response.argmax(axis=1)
            ]
            best_response[best_response == 0] = np.nan
        elif self.dim_a == 2:
            best_response = best_response.sum(axis=2 - axis_a)
            best_response_y = [
                self.a_discr[axis_a][a_idx] for a_idx in best_response.argmax(axis=1)
            ]
            best_response[best_response == 0] = np.nan

        ax.imshow(
            best_response.T / self.margin(),
            cmap="Wistia",
            vmin=0,
            vmax=1.1,
            alpha=0.7,
            extent=param_extent,
            origin="lower",
            aspect="auto",
        )
        ax.plot(
            self.o_discr,
            best_response_y,
            color="tab:orange",
            linewidth=2,
            label="best\nresponse",
        )

        # labels
        title_label = (
            f"Gradient - Agent {self.agent} "
            + (f", Bid {axis_a+1}" if self.dim_a > 1 else f"")
            + (
                ""
                if iter is None
                else (f" (Emp. Mean)" if iter == -1 else f" (Iteration {iter})")
            )
        )
        ax.set_title(
            title_label, fontsize=param["fontsize_title"], verticalalignment="bottom"
        )
        ax.set_xlabel("Observation o", fontsize=param["fontsize_label"])
        ax.set_ylabel("Bid b", fontsize=param["fontsize_label"])
        ax.legend(loc="upper left", fontsize=param["fontsize_legend"])

    def _plot_gradient_complete_info(
        self,
        ax,
        gradient: np.ndarray,
        param: dict,
        iter: int = None,
    ) -> None:
        """Plot Gradient (for standard incomplete information setting)

        Args:
            ax: axis from pyplot
            gradient (np.ndarray): gradient to visualize
            param (dict): specifies parameter (fontsize, ...) for plots
            iter (int, optional): show intermediate gradient. Defaults to None.
        """
        # plot strategy
        if self.dim_a > 1:
            raise NotImplementedError(
                "Visualization for complete information (n=1) case only implemented for dim_a = 1"
            )

        ax.bar(
            self.a_discr,
            gradient[0],
            color="gray",
            width=(self.a_discr[-1] - self.a_discr[0]) / self.m,
        )

        # plot best response
        index_br = gradient.argmax(axis=1).item()
        ax.axvline(
            x=self.a_discr[index_br],
            linestyle="--",
            color="tab:orange",
            linewidth=4,
            label="best\nresponse",
        )

        # labels
        title_label = (
            f"Gradient - Agent {self.agent}"
            + f" (o={self.o_discr[0]:.2f})"
            + (
                ""
                if iter is None
                else (f" (Emp. Mean)" if iter == -1 else f" (Iteration {iter})")
            )
        )
        ax.set_title(
            title_label, fontsize=param["fontsize_title"], verticalalignment="bottom"
        )
        ax.set_xlabel("Bid b", fontsize=param["fontsize_label"])
        ax.set_ylabel("Exp. Utility", fontsize=param["fontsize_label"])
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

    def _get_imshow_extent(self, axis_a: int) -> tuple:
        """determine extent for imshow for strategie and gradient plot"""
        delta_o = self.o_discr[1] - self.o_discr[0]
        if self.dim_a == 1:
            delta_a = self.a_discr[1] - self.a_discr[0]
            param_extent = (
                self.o_discr[0] - delta_o / 2,
                self.o_discr[-1] + delta_o / 2,
                self.a_discr[0] - delta_a / 2,
                self.a_discr[-1] + delta_a / 2,
            )
        else:
            delta_a = self.a_discr[axis_a][1] - self.a_discr[axis_a][0]
            param_extent = (
                self.o_discr[0] - delta_o / 2,
                self.o_discr[-1] + delta_o / 2,
                self.a_discr[axis_a][0] - delta_a / 2,
                self.a_discr[axis_a][-1] + delta_a / 2,
            )
        return param_extent

    # -------------------------------------- METHODS USED TO SAVE AND LOAD --------------------------------------- #

    def save(self, filename: str, save_init: bool = False):
        """Save strategy

        Args:
            filename (str): filename (incl. path) of strategy
            save_init (bool, optional): Save initial strategy. Defaults to False.
        """
        np.save(f"{filename}_agent_{self.agent}.npy", self.x)
        if save_init:
            np.save(f"{filename}_init.npy", self.history[0])

    def load(self, filename: str, load_init: bool = False) -> None:
        """Load saved strategy

        Args:
            filename (str): filename of strategy
            path (str): path to experiment directory
        """
        try:
            filename = f"{filename}_agent_{self.agent}" + (
                "_init.npy" if load_init else ".npy"
            )
            self.x = np.load(filename)
        except:
            print(f"File {filename} not found.")
            self.x = None

    def load_scale(self, name: str, path: str, n_scaled: int, m_scaled) -> None:
        """Get a upscaled version of a strategy

        Args:
            name (str): name of saved strategy
            path (str): path to experiment directory
            n_scaled (int): discretization (type) in the larger setting
            m_scaled (int): discretization (action) in the larger setting

        Raises:
            ValueError: _description_
            NotImplementedError: _description_
        """
        try:
            filename = os.path.join(path, name) + ".npy"
            strat = np.load(filename)
            bool_strat_loaded = True

        except:
            bool_strat_loaded = False
            print(f"File {filename} not found.")

        # only for two dimensional strategies
        if bool_strat_loaded:
            if len(self.x.shape) == 2:
                # idea: put probability measure of action two closest action in new discretization
                n, m = strat.shape
                a_discr = self.game.discr_interval(
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

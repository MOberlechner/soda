from time import sleep, time
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from soda.game import Game
from soda.learner.gradient import Gradient
from soda.mechanism.mechanism import Mechanism
from soda.strategy import Strategy


class Learner:
    """General learning class for approximation game.
    In each iteration methods are updated according to the specified learning algorithm (method)

    Implemented methods (subclasses) so far are:
        soda: Simultanoues Online Dual Averaging
        soma: Projected Online Mirror Ascent
        frank_wolfe: Frank-Wolfe Algorithm
        best_response: Best-Response Dynamic

    Attributes:
        method (str): specify learning algorithm
        max_iter (int): maximal number of iterations
        tol (float): stopping criterion (relative utility loss)
    """

    def __init__(
        self,
        max_iter: int,
        tol: float,
        stop_criterion: str = "util_loss",
        param: Dict = {},
    ):
        """Initialize learner

        Args:
            method (str): learning algorithm (soda, pogd, frank-wolfe,)
            max_iter (int): maximal number of iterations
            tol (float): stopping criterion (relative utility loss)
            stop_criterion (str): specify stopping criterion. Defaults to "util_loss"
        """
        self.name = "not defined"
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.stop_criterion = stop_criterion
        self.param = param

        self.convergence = False

    def __repr__(self) -> str:
        return f"Learner({self.name})"

    def __str__(self) -> str:
        general_info = f"Learner({self.name})\n- max_iter: {self.max_iter}\n- stop_criterion: {self.stop_criterion} < {self.tol}\n"
        print(self.param)
        if self.param != {}:
            str_parameter = f"- parameter:\n"
            for key, value in self.param.items():
                str_parameter += f"   - {key}: {value}\n"
        else:
            str_parameter = ""

        return general_info + str_parameter

    def run(
        self,
        game: Game,
        strategies: Dict[str, Strategy],
        disable_tqdm_bool: bool = True,
        print_result_bool: bool = False,
        save_history_bool: bool = False,
    ) -> None:
        """Run learning algorithm

        Args:
            game (class): discretized approximation game
            strategies (dict): strategy profile
            disable_tqdm_bool (bool): Disable progess bar. Defaults to True
            print_result_bool (bool): Print result. Defaults to False
            save_history_bool (bool): Save history of iterates (apart from first iteration). Defaults to False
        """

        # prepare gradients, i.e., compute path and indices
        self.gradient = Gradient()
        self.gradient.prepare(game, strategies)

        # prepare strategies
        for i in strategies:
            strategies[i].prepare_history(self.max_iter, save_history_bool)

        # init parameters
        min_max_value = 999
        t_max = self.max_iter - 1

        for t in tqdm(
            range(self.max_iter),
            unit_scale=True,
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}",
            disable=disable_tqdm_bool,
        ):

            # compute gradients
            for i in game.set_bidder:
                strategies[i].gradient = self.gradient.compute(game, strategies, i)

            # update history (utility, utility loss, dist_prev_iter, optional: strategy, gradient)
            for i in game.set_bidder:
                strategies[i].update_history(t)

            # check convergence
            min_max_value, max_value = self.check_convergence(
                t, strategies, min_max_value
            )
            if self.convergence:
                t_max = t
                break

            # update strategy
            for i in game.set_bidder:
                self.update_strategy(strategies[i], strategies[i].gradient, t)

        # print result
        if print_result_bool:
            self.print_result(self.convergence, min_max_value, max_value, t_max)

    def update_strategy(self, strategy: Strategy, gradient: np.ndarray, t: int):
        """Update strategy according to update rule from specific learning method

        Args:
            strategy (class):
            gradient (np.ndarray): gradient at iteration t
            t (int): teration (starting at 0)

        Each learner has the method update_strategy where the strategy is update according
        to a rule defined in the method update_step. This separation is done, to have the general update
        step from the optimization method independent of the strategy class.
        """
        raise NotImplementedError

    def check_convergence(
        self, t: int, strategies: dict, min_max_value: float
    ) -> tuple:
        """Check stopping criterion
        We check if the maximal (over all agents) "metric" is less than the tolerance
            - utility_loss: relative utility loss w.r.t. best response
            - dist_euclidean: Euclidean distance to previous iterate
            - dist_wasserstein: Wasserstein distance (mean over all mixed strategies) to previous iterate

        Args:
            t (int): current iteration
            strategies (dict): contains all strategies
            min_max_value (float): minimal value over all iterations of maximal value of stopping criterion over all agents

        Returns:
            float, float: new value of min_max_value, current value of metric
        """

        if self.stop_criterion == "util_loss":
            # update minimal max_util_loss (over all iterations)
            max_value = np.max([strategies[i].utility_loss[t] for i in strategies])

        elif self.stop_criterion == "dist_euclidean":
            # update minimal distance to last iterate (over all iterations)
            max_value = np.max([strategies[i].dist_prev_iter[t] for i in strategies])

        elif self.stop_criterion == "dist_wasserstein":
            raise NotImplementedError

        else:
            raise ValueError(
                f"Stopping critertion {self.stop_criterion} unknown. Choose util_loss or dist"
            )
        min_max_value = min(min_max_value, max_value)
        self.convergence = max_value < self.tol
        self.iter = t
        return min_max_value, max_value

    def print_result(
        self, convergence: bool, min_max_value: float, max_value: float, t_max: int
    ) -> None:
        """Print result of run

        Args:
            convergence (bool): did method converge
            min_max_value (float): best value of stopping critertion of worst agent
            max_value (float): current value of stopping critertion of worst agent
            t_max (int): number of iteration until convergence (0 if no convergence)
        """

        if convergence:
            print(f"Convergence after {t_max} iterations")
            print(
                f"Value of stopping criterion ({self.stop_criterion})",
                round(min_max_value, 5),
            )
        else:
            print("No convergence with stopping criterion")
            print(f"Current value of ({self.stop_criterion}): {max_value:.5f}")
            print(f"Best value of ({self.stop_criterion})   : {min_max_value:.5f})")


# Additional Functions used by several learning algorithms


def project_euclidean(x: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Projection w.r.t. Euclidean distance
    each row x[i] is projected to the probability simplex scaled by prior[i]
    Algorithm based on https://arxiv.org/pdf/1309.1541.pdf
    We allow for 1-dim observation space and 1- or 2-dim action space

    Args:
        x (np.ndarry):
        prior (np.ndarray): marginal prior

    Returns:
        np.ndarray: projection of x
    """
    bool_split_award = False
    if len(x.shape) > 2:
        # Split Award (1-dim observation space, 2-dim action space)
        if (len(prior.shape) == 1) & (len(x.shape) == 3):
            bool_split_award = True
            n, m1, m2 = x.shape
            x = x.reshape(n, m1 * m2)
        else:
            raise NotImplementedError(
                "Projection only implemented for 1-dim action space and valuation space"
            )

    n, m = x.shape

    assert n == len(prior), "dimensions of strategy and prior not compatible"

    # sort
    x_sort = -np.sort(-x, axis=1)
    x_cumsum = x_sort.cumsum(axis=1)
    # find rho
    rho = np.array(
        (x_sort + (prior.reshape(n, 1) - x_cumsum) / np.arange(1, m + 1) > 0).sum(
            axis=1
        ),
        dtype=int,
    )
    # define lambda
    lamb = 1 / rho * (prior - x_cumsum[range(n), rho - 1])
    x_proj = (x + np.repeat(lamb, m).reshape(n, m)).clip(min=0)

    # Split Award (1-dim observation space, 2-dim action space)
    if bool_split_award:
        return x_proj.reshape(n, m1, m2)
    else:
        return x_proj

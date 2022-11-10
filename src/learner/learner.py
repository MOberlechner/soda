from time import sleep, time
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.learner.gradient import Gradient


class Learner:
    """General learning class for approximation game.
    In each iteration methods are updated according to the specified learning algorithm (method)

    Implemented methods (subclasses) so far are:
        soda: dual averaging with entropic regularizer
        frank_wolfe: Frank-Wolfe Algorithm
        pogd: Projected Online Gradient Descent
        brgd: Best-Response-Gradient-Dynamic: Mixture of SODA and Frank-Wolfe
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
    ):
        """Initialize learner

        Args:
            method (str): learning algorithm (soda, pogd, frank-wolfe,)
            max_iter (int): maximal number of iterations
            tol (float): stopping criterion (relative utility loss)
            stop_criterion (str): specify stopping criterion. Defaults to "util_loss"
        """
        self.learner = "not defined"
        self.max_iter = max_iter
        self.tol = tol
        self.stop_criterion = stop_criterion

    def run(
        self,
        mechanism,
        game,
        strategies,
        disable_tqdm: bool = True,
        print: bool = False,
    ) -> None:
        """Run learning algorithm

        Args:
            mechanism (class): auction game
            game (class): discretized approximation game
            strategies (dict): strategy profile
            disable_tqdm (bool): Disable progess bar. Defaults to True
            print (bool): Print result. Defaults to False
        """

        # prepare gradients, i.e., compute path and indices
        gradient = Gradient()
        gradient.prepare(mechanism, game, strategies)

        # init parameters
        min_max_value = 999
        t_max = 0

        for t in tqdm(
            range(self.max_iter),
            unit_scale=True,
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}",
            disable=disable_tqdm,
        ):

            # compute gradients
            for i in game.set_bidder:
                gradient.compute(mechanism, game, strategies, i)

            # update history (strategy, gradient, utility, utility loss)
            for i in game.set_bidder:
                strategies[i].update_history(gradient.x[i])

            # check convergence
            convergence, min_max_value = self.check_convergence(
                strategies, min_max_value
            )
            if convergence:
                t_max = t
                break

            # update strategy
            for i in game.set_bidder:
                self.update_strategy(strategies[i], gradient.x[i], t)

        # print result
        if print:
            self.print_result(convergence, min_max_value, t_max, strategies)

    def update_strategy(self, strategy, gradient, t):
        """Update strategy according to update rule from specific learning method

        Args:
            strategy (class):
            gradient (np.ndarray): gradient at iteration t
            t (int): teration (starting at 0)

        Each learner has the method update_strategy where the strategy is update according
        to a rule defined in the method update_step. This separation is done, to have the general update
        step from the optimization method independent of the strategy class.
        """
        pass

    def check_convergence(self, strategies, min_max_value: float) -> Tuple:
        """Check stopping criterion
        We check if the maximal (over all agents) "metric" is less than the tolerance
        utility_loss: relative utility loss w.r.t. best response
        dist_euclidean: Euclidean distance to previous iterate
        dist_wasserstein: Wasserstein distance (mean over all mixed strategies) to previous iterate

        Args:
            strategies (dict): contains all strategies
            min_max_value (float): minimal value over all iterations of maximal value of stopping criterion over all agents

        Returns:
            Tuple: convergence (bool), new value of min_max_value (float)
        """
        if self.stop_criterion == "util_loss":
            # update minimal max_util_loss (over all iterations)
            max_util_loss = np.max([strategies[i].utility_loss[-1] for i in strategies])
            min_max_value = min(min_max_value, max_util_loss)

            # check if smaller than tolerance
            convergence = max_util_loss < self.tol

            return convergence, min_max_value

        elif self.stop_criterion == "dist_euclidean":
            # update minimal distance to last iterate (over all iterations)
            max_distance = np.max(
                [
                    np.linalg.norm(
                        strategies[i].history[-1] - strategies[i].history[-2]
                    )
                    if len(strategies[i].history) >= 2
                    else 1
                    for i in strategies
                ]
            )
            min_max_value = min(min_max_value, max_distance)

            # check if smaller than tolerance
            convergence = max_distance < self.tol

            return convergence, min_max_value

        elif self.stop_criterion == "dist_wasserstein":
            raise NotImplementedError

        else:
            raise ValueError(
                "Stopping critertion {} unknown. Choose util_loss or dist".format(
                    self.stop_criterion
                )
            )

    def print_result(
        self, convergence: bool, min_max_value: float, t_max: int, strategies
    ):
        """Print result of run

        Args:
            convergence (bool): did method converge
            min_max_value (float): best relative utility loss of worst agent
            t_max (int): number of iteration until convergence (0 if no convergence)
            strategies (class): current strategy profile
        """

        if convergence:
            print("Convergence after", t_max, "iterations")
            print(
                "Value of stopping criterion ({})".format(self.stop_criterion),
                round(min_max_value, 5),
            )
        else:
            max_util_loss = np.max([strategies[i].utility_loss[-1] for i in strategies])
            print("No convergence")
            print(
                "Current value of stopping criterion ({})".format(self.stop_criterion),
                round(max_util_loss, 5),
            )
            print(
                "Best value of stopping criterion ({})".format(self.stop_criterion),
                round(min_max_value, 5),
            )

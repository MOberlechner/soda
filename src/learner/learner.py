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
    ):
        """Initialize learner

        Args:
            method (str): learning algorithm (soda, pogd, frank-wolfe,)
            max_iter (int): maximal number of iterations
            tol (float): stopping criterion (relative utility loss)
        """
        self.learner = "not defined"
        self.max_iter = max_iter
        self.tol = tol

    def run(self, mechanism, game, strategies) -> None:
        """Run learning algorithm

        Args:
            mechanism (class): auction game
            game (class): discretized approximation game
            strategies (dict): strategy profile
        """

        # prepare gradients, i.e., compute path and indices
        gradient = Gradient()
        gradient.prepare(mechanism, game, strategies)

        # init parameters
        min_max_util_loss = 1
        t_max = 0

        for t in tqdm(
            range(self.max_iter),
            unit_scale=True,
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}",
        ):

            # compute gradients
            for i in game.set_bidder:
                gradient.compute(mechanism, game, strategies, i)

            # update history (strategy, gradient, utility, utility loss)
            for i in game.set_bidder:
                strategies[i].update_history(gradient.x[i])

            # check convergence
            convergence, min_max_util_loss = self.check_convergence(
                strategies, min_max_util_loss
            )
            if convergence:
                t_max = t
                break

            # update strategy
            for i in game.set_bidder:
                self.update_strategy(strategies[i], gradient.x[i], t)

        self.print_result(self, convergence, min_max_util_loss, t_max, strategies)

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

    def check_convergence(self, strategies, min_max_util_loss: float) -> Tuple:
        """Check if the maximal relative utility loss (over all agents) is less than the tolerance

        Args:
            strategies (dict): contains all strategies
            min_max_util_loss (float): previous value of min_max_util_loss

        Returns:
            Tuple: convergence (bool), new value of min_max_util_loss (float)
        """
        # update minimal max_util_loss (over all iterations)
        max_util_loss = np.max([strategies[i].utility_loss[-1] for i in strategies])
        min_max_util_loss = min(min_max_util_loss, max_util_loss)

        # check if smaller than tolerance
        convergence = max_util_loss < self.tol

        return convergence, min_max_util_loss

    def print_result(self, convergence, min_max_util_loss, t_max, strategies):
        """Print result of run

        Args:
            convergence (bool): did method converge
            min_max_util_loss (_type_): best relative utility loss of worst agent
            t_max (int): number of iteration until convergence (0 if no convergence)
            strategies (class): current strategy profile
        """

        if convergence:
            print("Convergence after", t_max, "iterations")
            print("Relative utility loss", round(min_max_util_loss * 100, 3), "%")
        else:
            max_util_loss = np.max([strategies[i].utility_loss[-1] for i in strategies])
            print("No convergence")
            print("Current relative utility loss", round(max_util_loss * 100, 3), "%")
            print("Best relative utility loss", round(min_max_util_loss * 100, 3), "%")

from sre_parse import State
from time import sleep, time
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.learner.gradient import Gradient


class FrankWolfe:
    """Frank Wolfe Algorithm applied to the discretized game

    Attributes:
        General
            max_iter (int): maximal number of iterations
            tol (float): algorithm stops if relative utility loss is less than tol
            grad (Dict): contains last computed gradient (np.ndarray) for each agent

        Steprule
            step_rule (bool): if True we 2/(t+1), else best response dynamic

        Gradient (opt_einsum)
            path (Dict): path for einsum to compute gradient for each agent
            indices (Dict): indices

    """

    def __init__(
        self,
        max_iter: int,
        tol: float,
        steprule_bool: bool,
    ):

        self.max_iter = max_iter
        self.tol = tol
        self.steprule_bool = steprule_bool

    def run(self, mechanism, game, strategies: Dict) -> None:
        """Runs SODA and updates strategies accordingly

        Args:
            mechanism: describes auction game
            game: discretized mechanism
            strategies (Dict): contains strategies for all agents
        """

        # prepare gradients, i.e., compute path and indices
        if not mechanism.own_gradient:
            gradient = Gradient()
            gradient.prepare(game, strategies)
        else:
            gradient = Gradient()

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
                if mechanism.own_gradient:
                    gradient.x[i] = mechanism.compute_gradient(strategies, game, i)
                else:
                    gradient.compute(strategies, game, i)

            # update utility, utility loss, history
            for i in game.set_bidder:
                strategies[i].update_utility(gradient.x[i])
                strategies[i].update_utility_loss(gradient.x[i])
                strategies[i].update_history()

            # check convergence
            convergence, min_max_util_loss = self.check_convergence(
                strategies, min_max_util_loss
            )
            if convergence:
                t_max = t
                break

            # update strategy
            for i in game.set_bidder:
                stepsize = self.step_rule(t)
                self.update_strategy(strategies[i], gradient.x[i], stepsize)

        # Print result
        if convergence:
            print("Convergence after", t_max, "iterations")
            print("Relative utility loss", round(min_max_util_loss * 100, 3), "%")
        else:
            max_util_loss = np.max([strategies[i].utility_loss[-1] for i in strategies])
            print("No convergence")
            print("Current relative utility loss", round(max_util_loss * 100, 3), "%")
            print("Best relative utility loss", round(min_max_util_loss * 100, 3), "%")

    def check_convergence(self, strategies, min_max_util_loss: float) -> Tuple:
        """check if the maximal relative utility loss (over all agents) is less than the tolerance"""
        max_util_loss = np.max([strategies[i].utility_loss[-1] for i in strategies])
        # update minimal max_util_loss (over all iterations)
        min_max_util_loss = min(min_max_util_loss, max_util_loss)

        # check if smaller than tolerance
        convergence = max_util_loss < self.tol

        return convergence, min_max_util_loss

    def update_strategy(
        self, strategy, gradient: np.ndarray, stepsize: np.ndarray
    ) -> None:
        """
        Update strategy: Frank-Wolfe Algorithm (or Best Response dynamic)

        Parameters
        ----------
        strategy : class Strategy
        gradient : np.ndarray,
        stepsize : np.ndarray, step size
        """
        # solution of minimization problem
        y = strategy.best_response(gradient)
        # make step towards y
        x_new = (1 - stepsize) * strategy.x + stepsize * y
        # update strategy
        strategy.x = x_new

    def step_rule(self, t: int) -> np.ndarray:
        """Compute step size:
        if step_rule is True: step sizes are not summable, but square-summable
        if step_rule is False: heuristic step size, scaled for each observation

        Args:
            t (int): current iteration

        Returns:
            np.ndarray: step size eta (either scaler or for each valuation)
        """

        if self.steprule_bool:
            # decreasing step rule: +2 since we start with t=0
            return np.array([2 / (t + 2)])

        else:
            # best response dynamic
            return np.array([1])

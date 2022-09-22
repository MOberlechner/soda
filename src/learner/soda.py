from time import sleep, time
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.learner.gradient import Gradient


class SODA:
    """Simultanoues Online Dual Averaging applied to discretized game

    Attributes:
        General
            max_iter (int): maximal number of iterations
            tol (float): algorithm stops if relative utility loss is less than tol
            grad (Dict): contains last computed gradient (np.ndarray) for each agent

        Steprule
            step_rule (bool): if True we use decreasing, else heuristic step size
            eta (float): parameter for step rule
            beta (float): parameter for step rule (if True)

        Gradient (opt_einsum)
            path (Dict): path for einsum to compute gradient for each agent
            indices (Dict): indices

    """

    def __init__(
        self,
        max_iter: int,
        tol: float,
        steprule_bool: bool,
        eta: float,
        beta: float = 1 / 20,
    ):

        self.max_iter = max_iter
        self.tol = tol
        self.steprule_bool = steprule_bool
        self.eta = eta
        self.beta = beta

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
                strategies[i].update_history_gradient(gradient.x[i])

            # check convergence
            convergence, min_max_util_loss = self.check_convergence(
                strategies, min_max_util_loss
            )
            if convergence:
                t_max = t
                break

            # update strategy
            for i in game.set_bidder:
                stepsize = self.step_rule(t, gradient.x[i], strategies[i].dim_o)
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

    def update_strategy(self, strategy, gradient: np.ndarray, stepsize: np.ndarray):
        """
        Update strategy: dual avering with entropic regularizer

        Parameters
        ----------
        strategy : class Strategy
        gradient : np.ndarray,
        stepsize : np.ndarray, step size

        Returns
        -------
        np.ndarray : updated strategy
        """

        # multiply with stepsize
        step = gradient * stepsize.reshape(list(stepsize.shape) + [1] * strategy.dim_a)
        xc_exp = strategy.x * np.exp(step)
        xc_exp_sum = xc_exp.sum(
            axis=tuple(range(strategy.dim_o, strategy.dim_o + strategy.dim_a))
        ).reshape(list(strategy.margin().shape) + [1] * strategy.dim_a)

        # update strategy
        strategy.x = (
            1
            / xc_exp_sum
            * strategy.prior.reshape(list(strategy.prior.shape) + [1] * strategy.dim_a)
            * xc_exp
        )

    def step_rule(self, t: int, grad: np.ndarray, dim_o: int) -> np.ndarray:
        """Compute step size:
        if step_rule is True: step sizes are not summable, but square-summable
        if step_rule is False: heuristic step size, scaled for each observation

        Args:
            t (int): current iteration
            grad (np.ndarray): gradient, necessary if steprule_bool is False (heuristic)
            dim_o (int): dimension of type space

        Returns:
            np.ndarray: step size eta (either scaler or for each valuation)
        """

        if self.steprule_bool:
            # decreasing step rule (Mertikopoulos&Zhou)
            return np.array(self.eta * 1 / (t + 1) ** self.beta)

        else:
            # heuristic
            scale = grad.max(axis=tuple(range(dim_o, len(grad.shape))))
            scale[scale < 0] = 0.001
            scale[scale < 1e-100] = 1e-100
            return self.eta / scale

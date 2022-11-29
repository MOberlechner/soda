import numpy as np

from .learner import Learner


class FrankWolfe(Learner):
    """Frank-Wolfe

    Attributes:
        method (str): specify learning algorithm
        max_iter (int): maximal number of iterations
        tol (float): stopping criterion (relative utility loss)
        stop_criterion (str): specify stopping criterion. Defaults to "util_loss"
    """

    def __init__(
        self,
        max_iter: int,
        tol: float,
        stop_criterion: str,
    ):
        super().__init__(max_iter, tol, stop_criterion)
        self.learner = "frank_wolfe"

    def update_strategy(self, strategy, gradient: np.ndarray, t: int) -> None:
        """
        Update strategy: Frank-Wolfe Algorithm (or Best Response dynamic)

        Parameters
        ----------
        strategy : class Strategy
        gradient : np.ndarray,
        stepsize : np.ndarray, step size
        """
        # solution of minimization problem
        br = strategy.best_response(gradient)
        # update step
        strategy.x = self.update_step(strategy.x, br, t)

    def update_step(self, x: np.ndarray, x_opt: np.ndarray, t):
        """Update Step for Frank-Wolfe

        Args:
            x (np.ndarray): current iterate
            x_opt (np.ndarray): solution of linear program
            stepsize (int): currrent iteration (starts at 0)
        """
        gamma = 2 / (t + 2)
        return (1 - gamma) * x + gamma * x_opt

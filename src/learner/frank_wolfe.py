import numpy as np

from .learner import Learner


class FrankWolfe(Learner):
    """Frank-Wolfe

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
        super().__init__(max_iter, tol)
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
        stepsize = self.step_rule(t)

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
        return np.array([2 / (t + 2)])

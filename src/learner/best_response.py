import numpy as np

from .learner import Learner


class BestResponse(Learner):
    """Best Response Dynamic

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
        self.learner = "best_response"

    def update_strategy(self, strategy, gradient: np.ndarray, t: int) -> None:
        """
        Update strategy: Best Response

        Parameters
        ----------
        strategy : class Strategy
        gradient : np.ndarray,
        stepsize : np.ndarray, step size
        """
        strategy.x = strategy.best_response(gradient)

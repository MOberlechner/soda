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
        stop_criterion: str,
    ):
        super().__init__(max_iter, tol, stop_criterion)
        self.name = "best_response"

    def update_strategy(self, strategy, gradient: np.ndarray, t: int) -> None:
        """_summary_

        Args:
            strategy (class): strategy class
            gradient (np.ndarray): gradient
            t (int): current iteration
        """
        strategy.x = strategy.best_response(gradient)

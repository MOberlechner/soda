import numpy as np

from .learner import Learner


class FictitiousPlay(Learner):
    """Fictitious Play

    Attributes:
        method (str): specify learning algorithm
        max_iter (int): maximal number of iterations
        tol (float): stopping criterion (relative utility loss)
    """

    def __init__(
        self,
        max_iter: int,
        tol: float,
        gradient_iter: int = -1,
    ):
        super().__init__(max_iter, tol)
        self.learner = "best_response"
        self.gradient_iter = gradient_iter

    def update_strategy(self, strategy, gradient: np.ndarray, t: int) -> None:
        """_summary_

        Args:
            strategy (class): strategy class
            gradient (np.ndarray): gradient
            t (int): current iteration
        """
        # gradient of empirical mean of play is equal to empirical mean of gradients
        if self.gradient_iter == -1:
            gradient_mean = np.mean(strategy.history_gradient, axis=0)

        else:
            gradient_mean = np.mean(
                strategy.history_gradient[-self.gradient_iter :], axis=0
            )

        strategy.x = strategy.best_response(gradient_mean)

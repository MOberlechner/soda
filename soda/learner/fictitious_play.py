import numpy as np
from src.learner.learner import Learner


class FictitiousPlay(Learner):
    """Fictitious Play
    Note that the current state of each agent (i.e. Strategy.x) is the empirical mean of historical play.
    This means we do not actually play the best response, but rather just update the empirical mean accordingly.

    Attributes:
        max_iter (int): maximal number of iterations
        tol (float): stopping criterion (relative utility loss)
        stop_criterion (str): specify stopping criterion. Defaults to 'util_loss'
    """

    def __init__(
        self, max_iter: int, tol: float, stop_criterion: str, param: dict = {}
    ):
        super().__init__(max_iter, tol, stop_criterion)
        self.name = "fictitious_play"

    def update_strategy(self, strategy, gradient: np.ndarray, t: int) -> None:
        """Update strategy according to Fictitious Play

        Args:
            strategy: Strategy
            gradient (np.ndarray): current gradient
            t (int): current iteration
        """
        br = strategy.best_response(gradient)
        strategy.x = self.update_step(strategy.x, br, t)

    def update_step(self, x: np.ndarray, br: np.ndarray, t: int) -> np.ndarray:
        """Update Step for Fictitious Play
        We store the empirical mean in the variable x

        Args:
            x (np.ndarray): current iterate
            br (np.ndarray): best response
            t (int): current iteration (starts at 0)
        """
        return 1 / (t + 2) * ((t + 1) * x + br)

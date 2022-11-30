import numpy as np

from src.learner.learner import Learner


class FrankWolfe(Learner):
    """Frank-Wolfe

    Attributes:
        method (str): specify learning algorithm
        max_iter (int): maximal number of iterations
        tol (float): stopping criterion (relative utility loss)
        stop_criterion (str): specify stopping criterion. Defaults to 'util_loss'
        method (str, optional): Choose between standard and online version of FW. Defaults to 'standard'.
        param_step (float, optional): parameter for step size for online version of FW
    """

    def __init__(
        self,
        max_iter: int,
        tol: float,
        stop_criterion: str,
        method: str = "standard",
        param_step: float = -0.2,
    ):
        super().__init__(max_iter, tol, stop_criterion)
        self.learner = "frank_wolfe"
        self.method = method
        self.param_step = param_step

    def update_strategy(self, strategy, gradient: np.ndarray, t: int) -> None:
        """Update strategy according to Frank-Wolfe (FW)

        Args:
            strategy: Strategy
            gradient (np.ndarray): current gradient
            t (int): current iteration
        """
        if self.method == "standard":
            br = strategy.best_response(gradient)
            step = 2 / (2 + t)

        elif self.method == "online":
            strategy.y += gradient
            gradient_agg = strategy.y / (t + 1)
            br = strategy.best_response(gradient_agg)
            step = (t + 1) ** self.param_step

        else:
            raise ValueError(f"'{self.method}'-Version of Frank-Wolfe unkonwn")

        strategy.x = self.update_step(strategy.x, br, step)

    def update_step(self, x: np.ndarray, x_opt: np.ndarray, step: float) -> np.ndarray:
        """Update Step for Frank-Wolfe

        Args:
            x (np.ndarray): current iterate
            x_opt (np.ndarray): solution of linear program
            step (float): step towards x_opt
        """
        return (1 - step) * x + step * x_opt

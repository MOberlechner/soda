import numpy as np

from .learner import Learner


class MultiplicativeWeights(Learner):
    """Multiplicative Weights

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
        param: dict,
    ):
        super().__init__(max_iter, tol, stop_criterion, param)
        self.check_input(param)

        self.name = "multiplicative_weights"
        self.stepsize = param["stepsize"]

    def update_strategy(self, strategy, gradient: np.ndarray, t: int) -> None:
        """
        Update strategy: Multiplicative Weights

        Parameters
        ----------
        strategy : class Strategy
        gradient : np.ndarray,
        stepsize : np.ndarray, step size
        """
        # update step
        strategy.x = self.update_step(
            strategy.x, gradient, strategy.prior, strategy.dim_o, strategy.dim_a
        )

    def update_step(
        self,
        x: np.ndarray,
        exp_util: np.ndarray,
        prior: np.ndarray,
        stepsize,
        dim_o: int,
        dim_a: int,
    ) -> np.ndarray:
        """_summary_

        Args:
            x (np.ndarray): current strategy
            exp_util (np.ndarray): expected utilities
            prior (np.ndarray): marginal prior distribution
            stepsize (_type_): stepsize
            dim_o (int): dimension of observation space
            dim_a (int): dimension of action space

        Returns:
            np.darray: updated strategy
        """
        x_upd = x * (1 + self.stepsize * exp_util)
        x_upd_sum = x_upd.sum(axis=tuple(range(dim_o, dim_o + dim_a))).reshape(
            list(prior.shape) + [1] * dim_a
        )
        return 1 / x_upd_sum * prior.reshape(list(prior.shape) + [1] * dim_a) * x_upd

    def check_input(self, param: dict) -> None:

        if "stepsize" not in param:
            raise ValueError("stepsize not defined in param")

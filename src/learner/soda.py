import numpy as np

from .learner import Learner


class SODA(Learner):
    """SODA - Dual Averaging with entropic regularizer

    Attributes:
        method (str): specify learning algorithm
        max_iter (int): maximal number of iterations
        tol (float): stopping criterion (relative utility loss)

        step_rule (bool): if True we use decreasing, else heuristic step size
        eta (float): factor for stepsize
        beta (float): rate the stepsize decreases with each iteration
    """

    def __init__(
        self,
        max_iter: int,
        tol: float,
        stop_criterion: str,
        steprule_bool: bool,
        eta: float,
        beta: float = 1 / 20,
    ):
        super().__init__(max_iter, tol, stop_criterion)
        self.learner = "soda"
        self.steprule_bool = steprule_bool
        self.eta = eta
        self.beta = beta

    def update_strategy(self, strategy, gradient: np.ndarray, t: int) -> None:
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
        # compute stepsize
        stepsize = self.step_rule(t, gradient, strategy.dim_o, strategy.dim_a)
        # make update step
        strategy.x = self.update_step(
            strategy.x,
            gradient,
            strategy.prior,
            stepsize,
            strategy.dim_o,
            strategy.dim_a,
        )

    def update_step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        prior: np.ndarray,
        eta_t: np.ndarray,
        dim_o: int = 1,
        dim_a: int = 1,
    ):
        """Update step for SODA, i.e. exponentiated gradient ascent or dual averaging with entropic regularizer

        Args:
            x (np.ndarray): current iterate
            grad (np.ndarray): gradient
            prior (np.ndarray): marginal prior distribution (scaling of prob. simplex)
            eta_t (np.ndarray): stepsize
            dim_o (int): dimension of type space
            dim_a (int): dimension of action space

        Returns:
            np.ndarray: updateted strategy
        """
        xc_exp = x * np.exp(grad * eta_t)
        xc_exp_sum = xc_exp.sum(axis=tuple(range(dim_o, dim_o + dim_a))).reshape(
            list(prior.shape) + [1] * dim_a
        )

        # update strategy
        return 1 / xc_exp_sum * prior.reshape(list(prior.shape) + [1] * dim_a) * xc_exp

    def step_rule(self, t: int, grad: np.ndarray, dim_o: int, dim_a: int) -> np.ndarray:
        """Compute step size:
        if step_rule is True: step sizes are not summable, but square-summable
        if step_rule is False: heuristic step size, scaled for each observation

        Args:
            t (int): current iteration
            grad (np.ndarray): gradient, necessary if steprule_bool is False (heuristic)
            dim_o (int): dimension of type space
            dim_a (int): dimension of action space

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
            stepsize = self.eta / scale
            return stepsize.reshape(list(stepsize.shape) + [1] * dim_a)

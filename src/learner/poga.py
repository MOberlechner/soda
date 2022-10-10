import numpy as np

from .learner import Learner


class POGA(Learner):
    """Projected Online Gradient Ascent (POGA)
    Basically Dual Averaging with euclidean regularizer

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
        steprule_bool: bool,
        eta: float,
        beta: float = 1 / 20,
    ):
        super().__init__(max_iter, tol)
        self.learner = "soda"
        self.steprule_bool = steprule_bool
        self.eta = eta
        self.beta = beta

    def update_strategy(self, strategy, gradient: np.ndarray, t: int) -> None:
        """Update strategy: Projected Gradient Ascent

        Args:
            strategy (class): strategy class
            gradient (np.ndarray): current gradient for agent i
            t (int): iteration
        """

        if strategy.dim_a > 1 or strategy.dim_o > 1:
            raise NotImplementedError(
                "Projection only implemented for 1-dim action space and valuation space"
            )
        # compute stepsize
        stepsize = self.step_rule(t, gradient, strategy.dim_o, strategy.dim_a)
        # make update step
        strategy.x = self.update_step(strategy.x, gradient, strategy.prior, stepsize)

    def update_step(
        self, x: np.ndarray, grad: np.ndarray, prior: np.ndarray, eta_t: np.ndarray
    ):
        """Update step Projected Online Gradient Ascent

        Args:
            x (np.ndarray): current iterate
            grad (np.ndarray): gradient
            prior (np.ndarray): marginal prior distribution (scaling of prob. simplex)
            eta_t (np.ndarray): step size at iteration t
        """
        return self.projection(x + grad * eta_t, prior)

    def projection(self, x: np.ndarray, prior: np.ndarray):
        """Projection w.r.t. Euclidean distance
        each row x[i] is projected to the probability simplex scaled by prior[i]
        Algorithm based on https://arxiv.org/pdf/1309.1541.pdf

        Args:
            x (np.ndarry): 2-dim array
            prior (np.ndarray): marginal prior

        Returns:
            np.ndarray: projection of x
        """
        n, m = x.shape
        assert n == len(prior), "dimensions of strategy and prior not compatible"

        # sort
        x_sort = -np.sort(-x, axis=1)
        x_cumsum = x_sort.cumsum(axis=1)
        # find rho
        rho = np.array(
            (x_sort + (prior.reshape(n, 1) - x_cumsum) / np.arange(1, m + 1) > 0).sum(
                axis=1
            ),
            dtype=int,
        )
        # define lambda
        lamb = 1 / rho * (prior - x_cumsum[range(n), rho - 1])
        return (x + np.repeat(lamb, m).reshape(n, m)).clip(min=0)

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

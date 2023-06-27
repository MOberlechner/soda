import numpy as np

from src.learner.learner import Learner
from src.util import learner_util


class SODA(Learner):
    """Simultaneous Online Dual Averaging

    Attributes:
        method (str): specify learning algorithm
        max_iter (int): maximal number of iterations
        tol (float): stopping criterion (relative utility loss)
        stop_criterion (str): specify stopping criterion. Defaults to "util_loss"

        regularizer (str): regularizer for dual averaging
        step_rule (bool): if True we use decreasing, else heuristic step size
        eta (float): factor for stepsize
        beta (float): rate the stepsize decreases with each iteration
    """

    def __init__(
        self,
        max_iter: int,
        tol: float,
        stop_criterion: str,
        param: dict,
    ):
        super().__init__(max_iter, tol, stop_criterion)
        self.check_input(param)

        self.name = "soda_" + param["regularizer"]
        self.steprule_bool = param["steprule_bool"]
        self.eta = param["eta"]
        self.beta = param["beta"]
        self.regularizer = param["regularizer"]

    def check_input(self):
        """Check Paramaters for Learner

        Raises:
            ValueError: regulizer unknown
        """
        if self.regularizer not in ["euclidean", "entropic"]:
            raise ValueError('Regularizer "{}" unknown'.format(self.regularizer))

    def update_strategy(self, strategy, gradient: np.ndarray, t: int) -> None:
        """Update Step for SODA

        Attributes:
            method (str): specify learning algorithm
            max_iter (int): maximal number of iterations
            tol (float): stopping criterion (relative utility loss)

            step_rule (bool): if True we use decreasing, else heuristic step size
            eta (float): factor for stepsize
            beta (float): rate the stepsize decreases with each iteration
        """
        # compute stepsize
        stepsize = self.step_rule(t, gradient, strategy.dim_o, strategy.dim_a)
        # make update step
        if self.regularizer == "euclidean":
            strategy.x, strategy.y = self.update_step_euclidean(
                strategy.y, gradient, stepsize, strategy.prior
            )
        elif self.regularizer == "entropic":
            strategy.x = self.update_step_entropic(
                strategy.x,
                gradient,
                stepsize,
                strategy.prior,
                strategy.dim_o,
                strategy.dim_a,
            )
        else:
            raise ValueError(f"regularizer {self.regularizer} unknown for soda")

    def update_step_euclidean(
        self,
        y: np.ndarray,
        grad: np.ndarray,
        eta_t: np.ndarray,
        prior: np.ndarray,
    ) -> tuple:
        """Update step SODA with euclidean mirror map, i.e. Lazy Projected Online Gradient Ascent

        Args:
            y (np.ndarray): current (dual) iterate
            grad (np.ndarray): gradient
            eta_t (np.ndarray): step size at iteration t
            prior (np.ndarray): marginal prior distribution (scaling of prob. simplex)

        Returns:
            tuple (np.ndarray): next dual and primal iterate
        """
        y += eta_t * grad
        x = learner_util.project_euclidean(y, prior)
        return x, y

    def update_step_entropic(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        eta_t: np.ndarray,
        prior: np.ndarray,
        dim_o: int = 1,
        dim_a: int = 1,
    ):
        """Update step SODA with entropic regularizer, i.e. Entropic Ascent Algorithm

        Args:
            x (np.ndarray): current iterate
            grad (np.ndarray): gradient
            eta_t (np.ndarray): stepsize
            prior (np.ndarray): marginal prior distribution (scaling of prob. simplex)
            dim_o (int): dimension of type space
            dim_a (int): dimension of action space

        Returns:
            np.ndarray: updateted strategy
        """
        xc_exp = x * np.exp(grad * eta_t)
        xc_exp_sum = xc_exp.sum(axis=tuple(range(dim_o, dim_o + dim_a))).reshape(
            list(prior.shape) + [1] * dim_a
        )
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

    def check_input(self, param: dict) -> None:
        """Check if all necessary parameters are in dict

        Args:
            param (dict): parameter for SODA algorithm

        Raises:
            ValueError: something is missing
        """
        for key in ["regularizer", "steprule_bool", "eta", "beta"]:
            if key not in param:
                raise ValueError(f"Define {key} in param")
        if param["regularizer"] not in ["entropic", "euclidean"]:
            raise ValueError("regularizer unkown")

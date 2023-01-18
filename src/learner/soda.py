import numpy as np

from .learner import Learner


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
        x = self.projection(y, prior)
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

    def projection(self, x: np.ndarray, prior: np.ndarray) -> np.ndarray:
        """Projection w.r.t. Euclidean distance
        each row x[i] is projected to the probability simplex scaled by prior[i]
        Algorithm based on https://arxiv.org/pdf/1309.1541.pdf

        Args:
            x (np.ndarry): 2-dim array
            prior (np.ndarray): marginal prior

        Returns:
            np.ndarray: projection of x
        """
        bool_split_award = False
        if len(x.shape) > 2:
            # Split Award (1-dim observation space, 2-dim action space)
            if (len(prior.shape) == 1) & (len(x.shape) == 3):
                bool_split_award = True
                n, m1, m2 = x.shape
                x = x.reshape(n, m1 * m2)
            else:
                raise NotImplementedError(
                    "Projection only implemented for 1-dim action space and valuation space"
                )

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
        x_proj = (x + np.repeat(lamb, m).reshape(n, m)).clip(min=0)

        # Split Award (1-dim observation space, 2-dim action space)
        if bool_split_award:
            return x_proj.reshape(n, m1, m2)
        else:
            return x_proj

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

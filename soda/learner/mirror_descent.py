import numpy as np

from soda.learner.learner import Learner, project_euclidean


class SOMA(Learner):
    """Simultaneous Online Mirror Ascent

    Attributes:
        method (str): specify learning algorithm
        max_iter (int): maximal number of iterations
        tol (float): stopping criterion (relative utility loss)
        stop_criterion (str): specify stopping criterion. Defaults to "util_loss"

        mirror_map (str): distance generating function for mirror ascent
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

        self.name = "soma_" + param["mirror_map"]
        self.mirror_map = param["mirror_map"]
        self.steprule_bool = param["steprule_bool"]
        self.eta = param["eta"]
        self.beta = param["beta"]

    def update_strategy(self, strategy, gradient: np.ndarray, t: int) -> None:
        """Update strategy: Projected Gradient Ascent

        Args:
            strategy (class): strategy class
            gradient (np.ndarray): current gradient for agent i
            t (int): iteration
        """
        # compute stepsize
        stepsize = self.step_rule(t, gradient, strategy.dim_o, strategy.dim_a)
        # make update step
        if self.mirror_map == "euclidean":
            strategy.x = self.update_step_euclidean(
                strategy.x, gradient, stepsize, strategy.prior
            )
        elif self.mirror_map == "entropic":
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
        x: np.ndarray,
        grad: np.ndarray,
        eta_t: np.ndarray,
        prior: np.ndarray,
    ) -> np.ndarray:
        """Update step SOMA with euclidean mirror map, i.e. Projected Online Gradient Ascent

        Args:
            x (np.ndarray): current iterate
            grad (np.ndarray): gradient
            prior (np.ndarray): marginal prior distribution (scaling of prob. simplex)
            eta_t (np.ndarray): step size at iteration t

        Returns:
            np.ndarray: next iterate
        """
        return project_euclidean(x + grad * eta_t, prior)

    def update_step_entropic(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        eta_t: np.ndarray,
        prior: np.ndarray,
        dim_o: int,
        dim_a: int,
    ) -> np.ndarray:
        """Update step SOMA with entropic mirror map, i.e. Entropic Ascent Algorithm

        Args:
            x (np.ndarray): current iterate
            grad (np.ndarray): gradient
            eta_t (np.ndarray): step size at iteration t
            prior (np.ndarray): marginal prior distribution (scaling of prob. simplex)
            dim_o (int): dimension observation space
            dim_a (int): dimension action space

        Returns:
            np.ndarray: next iterate
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

    def check_input(self, param: dict):
        """Check if all parameters are in dict

        Args:
            param (dict): parameter for learning algorithmus

        Raises:
            ValueError: mirror_map unkown
        """
        if "mirror_map" not in param:
            raise ValueError("define mirror_map in param")
        elif param["mirror_map"] not in ["euclidean", "entropic"]:
            raise ValueError("mirror map unknown")
        if "steprule_bool" not in param:
            raise ValueError("define steprule_bool in param")
        elif ("eta" not in param) or ("beta" not in param):
            raise ValueError("define eta and beta in param")

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
        method: str,
        steprule_bool: bool,
        eta: float,
        beta: float,
    ):
        super().__init__(max_iter, tol, stop_criterion)
        self.learner = "frank_wolfe"
        self.method = method
        self.steprule_bool = steprule_bool
        self.eta = eta
        self.beta = beta

        # check input
        if self.eta > 1:
            raise ValueError(f"eta={self.eta} not feasible (step > 1)")

    def update_strategy(self, strategy, gradient: np.ndarray, t: int) -> None:
        """Update strategy according to Frank-Wolfe (FW)

        Args:
            strategy: Strategy
            gradient (np.ndarray): current gradient
            t (int): current iteration
        """
        if self.method == "standard":
            br = strategy.best_response(gradient)
            eta_t = self.step_rule(t)

        elif self.method == "online":
            strategy.y += gradient
            gradient_agg = strategy.y / (t + 1)
            br = strategy.best_response(gradient_agg)
            eta_t = self.step_rule(t)

        else:
            raise ValueError(f"'{self.method}'-Version of Frank-Wolfe unkonwn")

        strategy.x = self.update_step(strategy.x, br, eta_t)

    def update_step(self, x: np.ndarray, x_opt: np.ndarray, step: float) -> np.ndarray:
        """Update Step for Frank-Wolfe

        Args:
            x (np.ndarray): current iterate
            x_opt (np.ndarray): solution of linear program
            step (float): step towards x_opt
        """
        return (1 - step) * x + step * x_opt

    def step_rule(self, t: int) -> np.ndarray:
        """Compute step size:
        if step_rule is True: step sizes are not summable, but square-summable
        if step_rule is False: standard rule for Frank-Wolfe (2/t+2)

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
            # standard FW
            return 2 / (t + 2)

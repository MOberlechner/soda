from time import sleep, time
from typing import Dict, List

import numpy as np
from opt_einsum import contract, contract_path
from tqdm import tqdm


class SODA:
    """Simultanoues Online Dual Averaging applied to discretized game

    Attributes:
        General
            max_iter (int): maximal number of iterations
            tol (float): algorithm stops if relative utility loss is less than tol
            grad (Dict): contains last computed gradient (np.ndarray) for each agent

        Steprule
            step_rule (bool): if True we use decreasing, else heuristic step size
            eta (float): parameter for step rule
            beta (float): parameter for step rule (if True)

        Gradient (opt_einsum)
            path (Dict): path for einsum to compute gradient for each agent
            indices (Dict): indices

    """

    def __init__(
        self,
        max_iter: int,
        tol: float,
        steprule_bool: bool,
        eta: float,
        beta: float = 1 / 20,
    ):

        self.max_iter = max_iter
        self.tol = tol
        self.steprule_bool = steprule_bool
        self.eta = eta
        self.beta = beta
        self.path = {}
        self.indices = {}
        self.grad = {}

    def run(self, mechanism, game, strategies: Dict) -> None:
        """Runs SODA and updates strategies accordingly

        Args:
            mechanism: describes auction game
            game: discretized mechanism
            strategies (Dict): contains strategies for all agents
        """

        # prepare gradients, i.e., compute path and indices
        if not mechanism.own_gradient:
            self.prepare_grad(game, strategies)
        # init variables
        convergence = False
        min_max_util_loss, max_util_loss = 1, 1
        t_max = 0

        for t in tqdm(
            range(self.max_iter),
            unit_scale=True,
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}",
        ):

            # compute gradients
            for i in game.set_bidder:
                if mechanism.own_gradient:
                    self.grad[i] = mechanism.compute_gradient(strategies, game, i)
                else:
                    self.grad[i] = self.compute_gradient(strategies, game, i)

            # update utility, utility loss, history
            for i in game.set_bidder:
                strategies[i].update_utility(self.grad[i])
                strategies[i].update_utility_loss(self.grad[i])
                strategies[i].update_history()

            # check convergence
            max_util_loss = np.max(
                [strategies[i].utility_loss[-1] for i in game.set_bidder]
            )
            min_max_util_loss = (
                max_util_loss
                if max_util_loss < min_max_util_loss
                else min_max_util_loss
            )
            if max_util_loss < self.tol:
                t_max = t
                convergence = True
                break

            # update strategy
            for i in game.set_bidder:
                stepsize = self.step_rule(t, self.grad[i], strategies[i].dim_o)
                strategies[i].update_strategy(self.grad[i], stepsize)

        # Print result
        if convergence:
            print("Convergence after", t_max, "iterations")
            print("Relative utility loss", round(max_util_loss * 100, 3), "%")
        else:
            print("No convergence")
            print("Current relative utility loss", round(max_util_loss * 100, 3), "%")
            print("Best relative utility loss", round(min_max_util_loss * 100, 3), "%")

    def step_rule(self, t: int, grad: np.ndarray, dim_o: int) -> np.ndarray:
        """Compute step size:
        if step_rule is True: step sizes are not summable, but square-summable
        if step_rule is False: heuristic step size, scaled for each observation

        Args:
            t (int): current iteration
            grad (np.ndarray): gradient, necessary if steprule_bool is False (heuristic)
            dim_o (int): dimension of type space

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
            return self.eta / scale

    def compute_gradient(self, strategies: Dict, game, agent: str) -> np.ndarray:
        """Computes gradient for agent given a strategyprofile, utilities (and weights)

        Args:
            strategies (Dict): contains strategies for agents
            game (_type_): approximation (discretized) game
            agent (str): specifies agent

        Returns:
            np.ndarray: gradient
        """

        opp = game.bidder.copy()
        opp.remove(agent)

        # bidders observations/valuations are independent
        if game.weights is None:
            return contract(
                self.indices[agent],
                *[game.utility[agent]]
                + [
                    strategies[i].x.sum(axis=tuple(range(strategies[i].dim_o)))
                    for i in opp
                ],
                optimize=self.path[agent]
            )
        # bidders observations/valuations are correlated
        else:
            return contract(
                self.indices[agent],
                *[game.utility[agent]]
                + [strategies[i].x for i in opp]
                + [game.weights],
                optimize=self.path[agent]
            )

    def prepare_grad(self, game, strategies: Dict) -> None:
        """Computes path and indices used in opt_einsum to compute gradients.
        Respective attributes are updated.

        Args:
            game: discretized game
            strategies (Dict): contains strategies for agents

        """

        ind = game.weights is None

        dim_o, dim_a = (
            strategies[game.bidder[0]].dim_o,
            strategies[game.bidder[0]].dim_a,
        )
        n_bidder = game.n_bidder

        # indices of all actions
        act = "".join([chr(ord("a") + i) for i in range(n_bidder * dim_a)])
        # indices of all valuations
        val = "".join([chr(ord("A") + i) for i in range(n_bidder * dim_o)])

        for i in game.set_bidder:

            idx = game.bidder.index(i)
            idx_opp = [i for i in range(n_bidder) if i != idx]

            # indices of utility array
            if game.private_values:
                # utility depends only on own oversvation
                start = act + val[idx * dim_o : (idx + 1) * dim_o]
            elif (not game.private_values) and (not ind):
                # utility depends on all observations
                start = act + val
            else:
                raise NotImplementedError

            # indices of bidder i's strategy
            end = (
                "->"
                + val[idx * dim_o : (idx + 1) * dim_o]
                + act[idx * dim_a : (idx + 1) * dim_a]
            )

            if ind:
                # valuations are independent
                self.indices[i] = (
                    start
                    + ","
                    + ",".join([act[j * dim_a : (j + 1) * dim_a] for j in idx_opp])
                    + end
                )
                self.path[i] = contract_path(
                    self.indices[i],
                    *[game.utility[i]]
                    + [
                        strategies[game.bidder[j]].x.sum(axis=tuple(range(dim_o)))
                        for j in idx_opp
                    ],
                    optimize="optimal"
                )[0]
            else:
                # valuations are correlated
                self.indices[i] = (
                    start
                    + ","
                    + ",".join(
                        [
                            val[j * dim_o : (j + 1) * dim_o]
                            + act[j * dim_a : (j + 1) * dim_a]
                            for j in idx_opp
                        ]
                    )
                    + ","
                    + val
                    + end
                )
                self.path[i] = contract_path(
                    self.indices[i],
                    *[game.utility[i]]
                    + [strategies[game.bidder[j]].x for j in idx_opp]
                    + [np.ones([game.n] * (dim_o * n_bidder))],
                    optimize="optimal"
                )[0]

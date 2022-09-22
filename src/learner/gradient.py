from typing import Dict, List

import numpy as np
from opt_einsum import contract, contract_path


class Gradient:
    """Class that handles the computation of the gradient for the discretized game

    Attributes
        General
            x (dict): current gradient for all agents

        Computation of Gradient (opt_einsum)
            indices (dict): indices used as input for einsum
            path (dict): used in opt_einsum to increase speed of computation

    """

    def __init__(self) -> None:
        self.x = {}
        self.path = {}
        self.indices = {}

    def compute(self, strategies: Dict, game, agent: str) -> None:
        """Computes gradient for agent given a strategyprofile, utilities (and weights)

        Args:
            strategies (Dict): contains strategies for agents
            game (_type_): approximation (discretized) game
            agent (str): specifies agent
            indices (Dict): contains str with indices

        Returns:
            np.ndarray: gradient
        """

        opp = game.bidder.copy()
        opp.remove(agent)

        # bidders observations/valuations are independent
        if game.weights is None:
            self.x[agent] = contract(
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
            self.x[agent] = contract(
                self.indices[agent],
                *[game.utility[agent]]
                + [strategies[i].x for i in opp]
                + [game.weights],
                optimize=self.path[agent]
            )

    def prepare(self, game, strategies: Dict) -> None:
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
        indices_act = "".join([chr(ord("a") + i) for i in range(n_bidder * dim_a)])
        # indices of all valuations
        indices_obs = "".join([chr(ord("A") + i) for i in range(n_bidder * dim_o)])

        for i in game.set_bidder:

            idx = game.bidder.index(i)
            idx_opp = [i for i in range(n_bidder) if i != idx]

            # indices of utility array
            if game.values == "private":
                # utility depends only on own oversvation
                start = indices_act + indices_obs[idx * dim_o : (idx + 1) * dim_o]

            elif game.values == "affiliated":
                # utility depends on all observations (affiliated values model)
                start = indices_act + indices_obs
            elif game.values == "common":
                # utility depends on common value, observations are independent (common value model)
                start = indices_act + "V"
            else:
                raise ValueError('value model "{}" unknown'.format(game.values))

            # indices of bidder i's strategy
            end = (
                "->"
                + indices_obs[idx * dim_o : (idx + 1) * dim_o]
                + indices_act[idx * dim_a : (idx + 1) * dim_a]
            )

            if ind:
                # valuations are independent
                self.indices[i] = (
                    start
                    + ","
                    + ",".join(
                        [indices_act[j * dim_a : (j + 1) * dim_a] for j in idx_opp]
                    )
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
                # indices for weights
                if game.values == "common":
                    indices_weight = "V" + indices_obs
                else:
                    indices_weight = indices_obs

                # valuations are correlated
                self.indices[i] = (
                    start
                    + ","
                    + ",".join(
                        [
                            indices_obs[j * dim_o : (j + 1) * dim_o]
                            + indices_act[j * dim_a : (j + 1) * dim_a]
                            for j in idx_opp
                        ]
                    )
                    + ","
                    + indices_weight
                    + end
                )
                self.path[i] = contract_path(
                    self.indices[i],
                    *[game.utility[i]]
                    + [strategies[game.bidder[j]].x for j in idx_opp]
                    + [game.weights],
                    optimize="optimal"
                )[0]

from typing import Dict, List

import numpy as np

from soda.game import Game
from soda.strategy import Strategy


class Gradient:
    """Class that handles the computation of the gradient for the discretized game

    Attributes
        Computation of Gradient (opt_einsum)
            indices (dict): indices used as input for einsum
            path (dict): used in opt_einsum to increase speed of computation

    """

    def __init__(self) -> None:
        self.path = {}
        self.indices = {}

    def compute(self, game: Game, strategies: Dict[str, Strategy], agent: str) -> None:
        """Computes gradient for agent given a strategyprofile, utilities (and weights)

        Args:
            game (Game): approximation game
            strategies (Dict[str, Strategy]): strategy profile
            agent (str): _description_
        """
        if game.mechanism.own_gradient:
            return game.mechanism.compute_gradient(game, strategies, agent)

        else:
            opp = game.bidder.copy()
            opp.remove(agent)

            # bidders observations/valuations are independent
            if game.weights is None:
                return np.einsum(
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
                return np.einsum(
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
            game (Game): discretized game
            strategies (Dict): contains strategies for agents

        """

        if game.mechanism.own_gradient:
            # nothing to do here
            pass

        else:
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
                if game.value_model == "private":
                    # utility depends only on own oversvation
                    start = indices_act + indices_obs[idx * dim_o : (idx + 1) * dim_o]

                elif game.value_model == "common_affiliated":
                    # utility depends on all observations (affiliated values model)
                    start = indices_act + indices_obs

                elif game.value_model == "common_independent":
                    # utility depends on common value, observations are independent (common value model)
                    start = indices_act + "V"
                else:
                    raise ValueError(
                        'value model "{}" unknown'.format(game.value_model)
                    )

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
                    self.path[i] = np.einsum_path(
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
                    if game.value_model == "common_independent":
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
                    self.path[i] = np.einsum_path(
                        self.indices[i],
                        *[game.utility[i]]
                        + [strategies[game.bidder[j]].x for j in idx_opp]
                        + [game.weights],
                        optimize="optimal"
                    )[0]

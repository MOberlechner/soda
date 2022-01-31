import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from itertools import product

from src.discretization import discr_spaces, init_strategy


class Strategy:

    def __init__(self, name, o_int: List, a_int: List, n: int, m: int, prior: str, init_method: str = 'random'):
        # name of the bidder
        self.name = name
        # action space and dimension of action space
        self.act, self.dim_a = discr_spaces(a_int, m)
        # observation space and dimension of observation space
        self.obs, self.dim_o = discr_spaces(o_int, n)
        # strategy
        self.x = init_strategy(init_method, prior)
        # utility
        self.utility, self.utility_loss = [], []
        self.history = [self.x]

    def margin(self):
        """
        Get marginal distribution over observations
        """
        return self.x.sum(axis=tuple(range(self.dim_o, self.dim_o + self.dim_a)))

    def bid(self, observation: np.ndarray):
        """
        Sample bids from the strategy

        Parameters
        ----------
        observation : np.ndarray, observations

        Returns
        -------
        np.ndarray, returns bids sampled from the respective mixed strategies given through the observations
        """
        # number of discretization points observations
        n = len(self.obs)

        # determine corresponding
        idx_obs = np.floor((observation - self.obs[0]) / (self.obs[-1] - self.obs[0]) * n).astype(int)
        idx_obs = np.maximum(0, np.minimum(n - 1, idx_obs))

        # sample bids from induced mixed strategy
        bids = np.array([np.random.choice(self.act, p=self.x[i]/self.x[i].sum()) for i in idx_obs])

        return bids

    # --------------------------------------- METHODS USED FOR COMPUTATION ------------------------------------------- #

    def best_response(self, gradient: np.ndarray):
        """

        Parameters
        ----------
        gradient : np.ndarray

        Returns
        -------
        np.ndarray, best response
        """
        # number of discretization points observations
        n = self.x.shape[0]
        # create array for best response
        best_response = np.zeros(gradient.shape)

        # determine largest entry of gradient for each valuation und put all the weight on the respective entry
        if self.dim_o == self.dim_a == 1:
            index_max = gradient.argmax(axis=1)
            best_response = best_response[range(n), index_max] = self.margin()
        else:
            for i in product(range(n), repeat=self.dim_o):
                index_max = np.unravel_index(np.argmax(gradient[i], axis=None), gradient[i].shape)
                best_response[i][index_max] = self.margin()[i]

        return best_response

    def update_strategy(self, gradient: np.ndarray, stepsize: np.ndarray):
        """
        Update strategy (exponentiated gradient)

        Parameters
        ----------
        gradient : np.ndarray,
        stepsize : np.ndarray, step size

        Returns
        -------

        """
        # multiply with stepsize
        step = gradient * stepsize.reshape(list(stepsize.shape) + [1] * self.dim_a)
        xc_exp = self.x * np.exp(step)
        xc_exp_sum = xc_exp.sum(axis=tuple(range(self.dim_o, self.dim_o + self.dim_a)))\
            .reshape(list(self.margin().shape) + [1]*self.dim_a)

        # update strategy
        self.x = 1 / xc_exp_sum * self.margin().reshape(list(self.margin().shape) + [1] * self.dim_a) * xc_exp

    def update_history(self):
        """
        Add current strategy in list
        """
        self.history += [self.x]

    def update_utility(self, gradient: np.ndarray):
        """
        Compute utility for current strategy and add to list self.utility
        """
        self.utility += [(self.x * gradient).sum()]

    def update_utility_loss(self, gradient: np.ndarray):
        """
        Compute relative utility loss for current strategy and add to list self.utility
        """
        self.utility_loss += [1 - (self.x * gradient).sum()/(self.best_response(gradient)*gradient).sum()]

    # --------------------------------------- METHODS USED TO ANALYZE RESULTS ---------------------------------------- #

    def plot(self):
        """
        Visualize current strategy
        """

        # parameters
        label_size = 13
        title_size = 14

        if self.dim_o == self.dim_a == 1:

            plt.imshow(self.x.T / self.margin(), extent=(self.obs[0], self.obs[-1], self.act[0], self.act[-1]),
                         origin='lower', vmin=0, cmap='Greys', aspect='auto')
            plt.ylabel('Bids', fontsize=label_size)
            plt.xlabel('Observations', fontsize=label_size)
            plt.title('Distributional Strategy Player \"' + self.name + '\"', fontsize=title_size)

        else:
            print('Plot for this strategy not available')



from itertools import product

import matplotlib.pyplot as plt
import numpy as np


class Strategy:
    def __init__(
        self,
        name,
        n: int,
        m: int,
        o_discr: np.ndarray,
        a_discr: np.ndarray,
        prior: np.ndarray,
    ):
        # name of the bidder
        self.name = name
        # observation and action space
        self.o_discr = o_discr
        self.a_discr = a_discr
        # dimension of spaces
        self.dim_o = 1 if len(self.o_discr.shape) == 1 else self.o_discr.shape[0]
        self.dim_a = 1 if len(self.a_discr.shape) == 1 else self.a_discr.shape[0]
        # prior (marginal) distribution
        self.prior = prior
        # strategy
        self.x = np.random.uniform(
            0, 1, size=tuple([n] * self.dim_o + [m] * self.dim_a)
        )
        # utility
        self.utility, self.utility_loss = [], []
        self.history = [self.x]

    def __str__(self):
        return "Strategy Bidder " + self.name + " - shape: " + str(self.x.shape)

    def margin(self):
        """
        Get marginal distribution over observations
        """
        return self.x.sum(axis=tuple(range(self.dim_o, self.dim_o + self.dim_a)))

    def initialize(
        self, init_method: str, param: dict = {}, lower_bound: float = 1e-50
    ):
        """
        Initializt strategy

        Parameters
        ----------
        init_method : strm specifies method such as equal, random, trufhul or function
        param : dict, used for method function, contains evaluation of function
        lower_bound : float,  used for method function

        Returns
        -------

        """

        if init_method == "random":
            sigma = np.random.uniform(0, 1, size=self.x.shape)

        elif init_method == "equal":
            sigma = np.ones(self.x.shape)

        elif init_method == "truthful":
            if self.dim_o == self.dim_a == 1:
                n, m = self.x.shape
                sigma = np.array(
                    [
                        [
                            np.exp(-1 / 2 * ((i - j) / 0.02) ** 2)
                            for j in np.linspace(0, 1, m)
                        ]
                        for i in np.linspace(0, 1, n)
                    ]
                )

        elif init_method == "function":
            b = param["init_function"]
            n, m = self.x.shape
            sigma = lower_bound * np.ones((n, m))
            for i in range(n):
                idx = (np.abs(self.a_discr - b[i])).argmin()
                sigma[i, idx] = 1

        else:
            raise ValueError("init_method not known")

        # normalize strategy according to prior
        sigma_sum = sigma.sum(axis=tuple(range(self.dim_o, self.dim_o + self.dim_a)))
        self.x = (1 / sigma_sum * self.prior).reshape(
            list(self.prior.shape) + [1] * self.dim_a
        ) * sigma

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
            best_response[range(n), index_max] = self.prior
        else:
            for i in product(range(n), repeat=self.dim_o):
                index_max = np.unravel_index(
                    np.argmax(gradient[i], axis=None), gradient[i].shape
                )
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
        xc_exp_sum = xc_exp.sum(
            axis=tuple(range(self.dim_o, self.dim_o + self.dim_a))
        ).reshape(list(self.margin().shape) + [1] * self.dim_a)

        # update strategy
        self.x = (
            1
            / xc_exp_sum
            * self.margin().reshape(list(self.margin().shape) + [1] * self.dim_a)
            * xc_exp
        )

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
        self.utility_loss += [
            1
            - (self.x * gradient).sum()
            / (self.best_response(gradient) * gradient).sum()
        ]

    # --------------------------------------- METHODS USED TO ANALYZE RESULTS ---------------------------------------- #

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
        n = self.x.shape[0]

        if self.dim_o == self.dim_a == 1:
            # determine corresponding
            idx_obs = np.floor(
                (observation - self.o_discr[0])
                / (self.o_discr[-1] - self.o_discr[0])
                * n
            ).astype(int)
            idx_obs = np.maximum(0, np.minimum(n - 1, idx_obs))

            # sample bids from induced mixed strategy (old, slower version)
            # bids = np.array([np.random.choice(self.a_discr, p=self.x[i]/self.x[i].sum()) for i in idx_obs])

            uniques, counts = np.unique(
                idx_obs, return_inverse=False, return_counts=True
            )
            bids = np.zeros(idx_obs.shape)
            for d, c in zip(uniques, counts):
                bids[idx_obs == d] = np.random.choice(
                    a_discr, size=c, p=sigma[d] / sigma[d].sum()
                )
            return bids

        else:
            raise NotImplementedError

    def plot(self):
        """
        Visualize current strategy
        """
        # parameters
        label_size = 13
        title_size = 14

        if self.dim_o == self.dim_a == 1:

            plt.imshow(
                self.x.T / self.margin(),
                extent=(
                    self.o_discr[0],
                    self.o_discr[-1],
                    self.a_discr[0],
                    self.a_discr[-1],
                ),
                origin="lower",
                vmin=0,
                cmap="Greys",
                aspect="auto",
            )
            plt.ylabel("bids b", fontsize=label_size)
            plt.xlabel("observations v", fontsize=label_size)
            plt.title(
                'Distributional Strategy Player "' + self.name + '"',
                fontsize=title_size,
            )

        else:
            print("Plot for this strategy not available")
            raise NotImplementedError

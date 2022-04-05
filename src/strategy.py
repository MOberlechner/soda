from itertools import product

import matplotlib.pyplot as plt
import numpy as np


class Strategy:
    def __init__(self, agent, game):
        # name of the bidder
        self.agent = agent
        # observation and action space
        self.o_discr = game.o_discr[agent]
        self.a_discr = game.a_discr[agent]
        # number of discretization points
        self.n = len(self.o_discr)
        self.m = len(self.a_discr)
        # dimension of spaces
        self.dim_o = 1 if len(self.o_discr.shape) == 1 else self.o_discr.shape[0]
        self.dim_a = 1 if len(self.a_discr.shape) == 1 else self.a_discr.shape[0]
        # prior (marginal) distribution
        self.prior = game.prior[agent]
        # strategy
        self.x = np.ones(tuple([game.n] * self.dim_o + [game.m] * self.dim_a)) / (
            game.n ** self.dim_o * game.m * self.dim_a
        )
        # utility
        self.utility, self.utility_loss = [], []
        self.history = []

    def __str__(self):
        return "Strategy Bidder " + self.agent + " - shape: " + str(self.x.shape)

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

        elif init_method == "truthful_inv":
            if self.dim_o == self.dim_a == 1:
                n, m = self.x.shape
                sigma = np.array(
                    [
                        [
                            np.exp(-1 / 2 * ((i - j) / 0.02) ** 2)
                            for j in np.linspace(0, 1, m)
                        ]
                        for i in np.linspace(1, 0, n)
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

        # save initial strategy in history
        self.update_history()

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

    def update_strategy(
        self, gradient: np.ndarray, stepsize: np.ndarray, method: str = "dual_averaging"
    ):
        """
        Update strategy (exponentiated gradient)

        Parameters
        ----------
        gradient : np.ndarray,
        stepsize : np.ndarray, step size
        method : str, allows us to switch between dual averaging and best response

        Returns
        -------

        """
        if method == "dual_averaging":
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
                * self.prior.reshape(list(self.prior.shape) + [1] * self.dim_a)
                * xc_exp
            )

        elif method == "best_response":
            self.x = self.best_response(gradient)

        else:
            raise ValueError("Error in update_strategy: method unknown")

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
        Add 1e-50 so that we don't divide by zero
        """
        self.utility_loss += [
            np.abs(
                1
                - (self.x * gradient).sum()
                / ((self.best_response(gradient) * gradient).sum() + 1e-50)
            )
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
                    self.a_discr, size=c, p=self.x[d] / self.x[d].sum()
                )
            return bids

        else:
            raise NotImplementedError

    def plot(self, more: bool = False, beta: np.ndarray = None):
        """
        Visualize current strategy

        more : bool, if true, we also plot utility loss and distance to last iterate

        """
        # parameters
        label_size = 13
        title_size = 14

        if more:

            # compute distance to last iterate
            dist = [np.linalg.norm(s - self.x) for s in self.history[:-1]]
            # plot utility loss and distance (utility loss plot maximal until 100%)
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 2)
            plt.plot(np.minimum(self.utility_loss, 1), label="utility loss", color="k")
            plt.plot(dist, label="dist. last iterate", color="k", linestyle="--")
            plt.yscale("log")
            plt.grid(axis="y")
            plt.legend()

            # create subplit for strategy
            plt.subplot(1, 2, 1)

        if self.dim_o == self.dim_a == 1:

            # plot strategy
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
                'Distributional Strategy Player "' + self.agent + '"',
                fontsize=title_size,
            )

            # plot BNE
            if beta is not None:
                x = np.linspace(self.o_discr[0], self.o_discr[-1], len(beta))
                plt.plot(x, beta, linestyle="--", color="r", label="analyt. BNE")
                plt.legend()

            plt.show()

        else:
            print("Plot for this strategy not available")
            raise NotImplementedError

    def save(self, name: str, path: str):
        """ Saves strategy in respective directory

        Parameters
        ----------
        name : str, name of strategy (without ending .npy)
        path : str ,path to directory ( do not append strategies/)
        """
        np.save(path + "strategies/" + name + "_agent_" + self.agent + ".npy", self.x)

    def load(self, name: str, path: str):
        try:
            self.x = np.load(
                path + "strategies/" + name + "_agent_" + self.agent + ".npy"
            )
        except:
            print(
                'File: "'
                + name
                + "_agent_"
                + self.agent
                + ".npy"
                + '" is not available in directory "'
                + path
                + "strategies/"
                + '"'
            )

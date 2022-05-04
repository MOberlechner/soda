import numpy as np
from scipy.stats import norm, powerlaw, uniform


def marginal_prior_pdf(mechanism, obs: np.ndarray, agent: str):

    if mechanism.prior == "uniform":
        return uniform.pdf(
            obs,
            loc=mechanism.o_space[agent][0],
            scale=mechanism.o_space[agent][-1] - mechanism.o_space[agent][0],
        )

    if mechanism.prior == "uniform_bi":
        eta = 0.9
        return eta * uniform.pdf(
            obs,
            loc=mechanism.o_space[agent][0],
            scale=mechanism.o_space[agent][-1] - mechanism.o_space[agent][0],
        ) + (1 - eta) * uniform.pdf(
            obs,
            loc=0.4 * (mechanism.o_space[agent][0] + mechanism.o_space[agent][-1]),
            scale=0.2 * (mechanism.o_space[agent][-1] - mechanism.o_space[agent][0]),
        )

    elif mechanism.prior == "gaussian":
        return norm.pdf(
            obs, loc=mechanism.param_prior["mu"], scale=mechanism.param_prior["sigma"]
        )

    elif mechanism.prior == "gaussian_bimodal":
        eta = 0.5
        return eta * norm.pdf(obs, loc=0.25, scale=0.1) + (1 - eta) * norm.pdf(
            obs, loc=0.75, scale=0.1
        )

    elif mechanism.prior == "powerlaw":
        power = mechanism.param_prior["power"][agent]
        return powerlaw.pdf(
            obs,
            a=power,
            loc=mechanism.o_space[agent][0],
            scale=mechanism.o_space[agent[0]][-1] - mechanism.o_space[agent][0],
        )

    elif mechanism.prior == "affiliated_values":
        # we assume that the observations are modeled as sum of two rv ~ U([0,1])
        return np.where(obs <= 1, obs, 2 - obs)


def compute_weights(game, mechanism):

    if "corr" in mechanism.param_prior:
        if mechanism.prior == "uniform":
            if mechanism.n_bidder == 2:
                # correlated prior with Bernoulli Paramater (according to Ausubel & Baranov 2020)
                gamma = mechanism.param_prior["corr"]
                weights = np.ones((game.n, game.n)) * (1 - gamma) * 1 / game.n
                weights[np.diag_indices(game.n, ndim=2)] = (
                    gamma * 1 + (1 - gamma) * 1 / game.n
                )
                weights = weights * game.n

            elif mechanism.name == "llg_auction":
                # correlated prior with Bernoulli Paramater (according to Ausubel & Baranov 2020, 2 players correlated, 1 player uncorrelated)
                gamma = mechanism.param_prior["corr"]
                weights = np.ones((game.n, game.n)) * (1 - gamma) * 1 / game.n
                weights[np.diag_indices(game.n, ndim=2)] = (
                    gamma * 1 + (1 - gamma) * 1 / game.n
                )
                weights = weights * game.n
                weights = np.repeat(weights, game.n).reshape(tuple([game.n] * 3))

            else:
                raise NotImplementedError(
                    "Correlation not implemented for this setting"
                )

            return weights

        else:
            raise NotImplementedError("Correlation only for uniform prior implemented")

    elif mechanism.prior == "affiliated_values":
        agent = game.set_bidder[0]
        common_prior = np.ones((game.n, game.n))
        for i in range(game.n):
            common_prior[i] = [
                prior_pdf_affiliated_values(
                    game.o_discr[agent][i], game.o_discr[agent][j]
                )
                for j in range(game.n)
            ]
        common_prior = common_prior / common_prior.sum()

        weights = common_prior / (
            common_prior.sum(axis=0)
            .reshape(game.n, 1)
            .dot(common_prior.sum(axis=1).reshape(1, game.n))
        )
        return weights

    else:
        return None


def prior_pdf_affiliated_values(x, y):
    """density function of affialiated values model (Krishna, Ex. 6.1) for two symmetric bidders"""
    if x <= 1:
        if y <= 1:
            return x if y > x else y
        if y > 1:
            return 0 if y > x + 1 else 1 + x - y
    else:
        if y <= 1:
            return 1 + y - x if y > x - 1 else 0
        if y > 1:
            return 2 - y if y > x else 2 - x

import numpy as np


def project_euclidean(x: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Projection w.r.t. Euclidean distance
    each row x[i] is projected to the probability simplex scaled by prior[i]
    Algorithm based on https://arxiv.org/pdf/1309.1541.pdf
    We allow for 1-dim observation space and 1- or 2-dim action space

    Args:
        x (np.ndarry):
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

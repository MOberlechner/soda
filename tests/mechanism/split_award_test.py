"""
This module tests the single-item mechanism
"""
import numpy as np
import pytest

from src.mechanism.split_award import SplitAwardAuction


@pytest.fixture
def get_mechanism():
    """
    create mechanism
    """
    bidder = ["1", "1"]
    o_space = {"1": [1, 1.4]}
    a_space = {"1": [[0.0, 2.5], [0.0, 2.5]]}
    param_prior = {"distribution": "uniform"}
    param_util = {
        "tie_breaking": "random",
        "payment_rule": "first_price",
        "scale": 0.3,
    }
    return SplitAwardAuction(bidder, o_space, a_space, param_prior, param_util)


def test_create_mechamism(get_mechanism):
    """
    Test if single-item mechanism can be created
    """
    assert isinstance(
        get_mechanism, SplitAwardAuction
    ), "create single-item auction mechanism"


def test_get_allocation(get_mechanism):
    """
    Test allocation method of split-award auction
    split-allocations are prefered over single-allocations

    """
    mechanism = get_mechanism
    bids = np.array(
        [
            [[1, 3, 4, 2], [1, 1, 2, 2]],
            [
                [2, 4, 3, 2],
                [1, 2, 2, 2],
            ],
        ]
    )

    mechanism.tie_breaking = "random"
    allocation = mechanism.get_allocation(bids, 0)

    assert np.array_equal(
        allocation[0], [1.0, 0.0, 0.0, 0.5]
    ), "allocation single-award, with tie-breaking"

    assert np.array_equal(
        allocation[1], [0.0, 1.0, 0.0, 0.0]
    ), "allocation split-award, with tie-breaking"

    mechanism.tie_breaking = "lose"
    allocation = mechanism.get_allocation(bids, 0)
    assert np.array_equal(
        allocation[0], [1.0, 0.0, 0.0, 0.0]
    ), "allocation single-award, with tie-breaking"
    assert np.array_equal(
        allocation[1], [0.0, 1.0, 0.0, 0.0]
    ), "allocation split-award, with tie-breaking"


def test_get_payment(get_mechanism):
    """
    Function to test payment method in split-award auction
    """
    mechanism = get_mechanism
    bids = np.array(
        [
            [[2, 3, 4, 2], [1, 2, 2, 2]],
            [
                [3, 5, 3, 2],
                [1, 2, 2, 2],
            ],
        ]
    )
    allocation = mechanism.get_allocation(bids, 0)

    mechanism.payment_rule = "first_price"
    payments = mechanism.get_payment(bids, allocation, 0)
    assert np.array_equal(payments[0], [0.0, 3.0, 0.0, 2.0]), "first_price payment rule"


def test_utility(get_mechanism):
    """
    Function to test utility method in split-award auction
    """
    mechanism = get_mechanism
    bids = np.array(
        [
            [[2, 3, 4, 2], [1, 2, 2, 2]],
            [
                [3, 5, 4, 2],
                [2, 2, 2, 2],
            ],
        ]
    )
    obs = np.array([1, 1, 1, 1])
    utility = mechanism.utility(obs, bids, 0)
    assert np.array_equal(
        utility, [1.0, 2.0, 1.7, 0.5]
    ), "utility for first_price payment rule and tie_breaking"

"""
This module tests the single-item mechanism
"""
import numpy as np
import pytest
from scipy.stats import pearsonr

from soda.mechanism.llg_auction import LLGAuction


@pytest.fixture
def get_mechanism():
    """
    Function to create mechanism
    """
    bidder = ["L", "L", "G"]
    o_space = {"L": [0, 1], "G": [0, 2]}
    a_space = {"L": [0, 1], "G": [0, 2]}
    param_prior = {
        "distribution": "uniform",
        "corr": 0.5,
    }
    param_util = {
        "payment_rule": "nearest_zero",
        "tie_breaking": "local",
    }
    return LLGAuction(bidder, o_space, a_space, param_prior, param_util)


def test_create_mechamism(get_mechanism):
    """
    Test if llg-auction mechanism can be created
    """
    mechanism = get_mechanism
    assert isinstance(mechanism, LLGAuction), "create llg-auction mechanism"


def test_get_allocation(get_mechanism: LLGAuction):
    """
    Test allocation method of llg-auction
    """
    mechanism = get_mechanism
    bids = np.array(
        [
            [0, 0, 1, 2, 1],
            [0, 2, 1, 0, 3],
            [0, 1, 2, 3, 4],
        ]
    )

    mechanism.tie_breaking = "random"
    allocation_L = mechanism.get_allocation(bids, 0)
    allocation_G = mechanism.get_allocation(bids, 2)
    assert np.array_equal(
        allocation_L, [0.5, 1, 0.5, 0, 0.5]
    ), "allocation local bidder, tie-breaking random"
    assert np.array_equal(
        allocation_G, [0.5, 0, 0.5, 1, 0.5]
    ), "allocation local bidder, tie-breaking random"

    mechanism.tie_breaking = "lose"
    allocation_L = mechanism.get_allocation(bids, 0)
    allocation_G = mechanism.get_allocation(bids, 2)
    assert np.array_equal(
        allocation_L, [0.0, 1, 0.0, 0.0, 0.0]
    ), "allocation local bidder, tie-breaking lose"
    assert np.array_equal(
        allocation_G, [0.0, 0, 0.0, 1.0, 0.0]
    ), "allocation local bidder, tie-breaking lose"

    mechanism.tie_breaking = "local"
    allocation_L = mechanism.get_allocation(bids, 0)
    allocation_G = mechanism.get_allocation(bids, 2)
    assert np.array_equal(
        allocation_L, [1.0, 1.0, 1.0, 0.0, 1.0]
    ), "allocation local bidder, tie-breaking lose"
    assert np.array_equal(
        allocation_G, [0.0, 0.0, 0.0, 1.0, 0.0]
    ), "allocation local bidder, tie-breaking lose"


def test_get_payment(get_mechanism: LLGAuction):
    """
    Test different payment methods in llg-auction (example for bids from Ausubel and Baranov 2020)
    """
    mechanism = get_mechanism
    bids = np.array(
        [
            [8, 12, 5],
            [6, 6, 4],
            [10, 10, 10],
        ]
    )
    allocation_L1 = mechanism.get_allocation(bids, 0)
    allocation_L2 = mechanism.get_allocation(bids, 1)
    allocation_G = mechanism.get_allocation(bids, 2)

    assert np.array_equal(
        mechanism.get_payment(bids, allocation_G, 2), [0, 0, 9]
    ), "global bidder,payment rule"

    mechanism.payment_rule = "nearest_zero"
    assert np.array_equal(
        mechanism.get_payment(bids, allocation_L1, 0), [5, 5, 0]
    ), "local bidder 1, nearest-zero payment rule"
    assert np.array_equal(
        mechanism.get_payment(bids, allocation_L2, 1), [5, 5, 0]
    ), "local bidder 2, nearest-zero payment rule"

    mechanism.payment_rule = "nearest_bid"
    assert np.array_equal(
        mechanism.get_payment(bids, allocation_L1, 0), [6, 8, 0]
    ), "local bidder 1, nearest-bid payment rule"
    assert np.array_equal(
        mechanism.get_payment(bids, allocation_L2, 1), [4, 2, 0]
    ), "local bidder 2, nearest-bid payment rule"

    mechanism.payment_rule = "nearest_vcg"
    assert np.array_equal(
        mechanism.get_payment(bids, allocation_L1, 0), [6, 7, 0]
    ), "local bidder 1, nearest-vcg payment rule"
    assert np.array_equal(
        mechanism.get_payment(bids, allocation_L2, 1), [4, 3, 0]
    ), "local bidder 2, nearest-vcg payment rule"

    mechanism.payment_rule = "first_price"
    assert np.array_equal(
        mechanism.get_payment(bids, allocation_L1, 0), [8, 12, 0]
    ), "local bidder 1, first-price payment rule"
    assert np.array_equal(
        mechanism.get_payment(bids, allocation_L2, 1), [6, 6, 0]
    ), "local bidder 2, first-price payment rule"
    assert np.array_equal(
        mechanism.get_payment(bids, allocation_G, 2), [0, 0, 10]
    ), "global bidder, first-price payment rule"


def test_sample_types_uniform(get_mechanism: LLGAuction):
    """
    Test adapted sample_types_uniform for llg-auction
    """
    mechanism = get_mechanism

    for corr in [0.0, 0.3, 0.5, 0.9]:
        mechanism.corr = corr
        obs = mechanism.sample_types_uniform(int(1e5))
        assert (
            pytest.approx(pearsonr(obs[0], obs[1])[0], 0.05) == 0.5
        ), f"sample uniform observations with corr={corr}"

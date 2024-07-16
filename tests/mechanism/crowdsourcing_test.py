"""
This module tests the crowdsourcing contests
"""
from typing import List

import numpy as np
import pytest

from soda.game import Game
from soda.learner.gradient import Gradient
from soda.mechanism.crowdsourcing import Crowdsourcing
from soda.strategy import Strategy


@pytest.fixture
def get_mechanism():
    """
    Function to create mechanism
    """

    def _mechanism(
        n_bidder: int, prices: List[float], tie_breaking: str, type: str
    ) -> Crowdsourcing:
        """
        create mechanism
        """
        bidder = ["1"] * n_bidder
        o_space = {"1": [0.0, 1.0]}
        a_space = {"1": [0.0, 1.0]}
        param_prior = {"distribution": "uniform"}
        param_util = {
            "prices": prices,
            "tie_breaking": tie_breaking,
            "payment_rule": "first_price",
            "type": type,
        }
        return Crowdsourcing(bidder, o_space, a_space, param_prior, param_util)

    return _mechanism


def test_create_mechamism(get_mechanism):
    """
    Test if contest mechanism can be created
    """
    mechanism = get_mechanism(
        n_bidder=3, prices=[0.8, 0.2, 0.0], tie_breaking="random", type="value"
    )
    assert isinstance(mechanism, Crowdsourcing), "create crowdsourcing contest"


def test_get_allocation_random(get_mechanism):
    """
    Test allocation method of crowdsourcing contest with random tie-breaking rule
    """
    mechanism = get_mechanism(
        n_bidder=3, prices=[0.8, 0.2, 0.0], tie_breaking="random", type="value"
    )
    bids = np.array([[0, 1, 2, 0], [0, 1, 1, 1], [0, 0, 1, 2]])
    result = np.array(
        [
            [[1 / 3, 0.5, 1.0, 0.0], [1 / 3, 0.5, 0.0, 0.0], [1 / 3, 0.0, 0.0, 1.0]],
            [[1 / 3, 0.5, 0.0, 0.0], [1 / 3, 0.5, 0.5, 1.0], [1 / 3, 0.0, 0.5, 0.0]],
            [[1 / 3, 0.0, 0.0, 1.0], [1 / 3, 0.0, 0.5, 0.0], [1 / 3, 1.0, 0.5, 0.0]],
        ]
    )
    # test allocation
    for idx in range(3):
        allocation = mechanism.get_allocation(bids, idx)
        assert isinstance(allocation, np.ndarray), "allocation returns array"
        assert np.allclose(allocation, result[idx]), f"allocation agent {idx}"


def test_get_allocation_lose(get_mechanism):
    """
    Test allocation method of crowdsourcing contest with random tie-breaking rule
    """
    mechanism = get_mechanism(
        n_bidder=3, prices=[0.8, 0.2, 0.0], tie_breaking="lose", type="value"
    )
    bids = np.array([[0, 1, 2, 0], [0, 1, 1, 1], [0, 0, 1, 2]])
    result = np.array(
        [
            [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 1]],
            [[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
            [[0, 0, 0, 1], [0, 0, 0, 0], [1, 1, 1, 0]],
        ]
    )
    # test allocation
    for idx in range(3):
        allocation = mechanism.get_allocation(bids, idx)
        assert isinstance(allocation, np.ndarray), "allocation returns array"
        assert np.allclose(allocation, result[idx]), f"allocation agent {idx}"


def test_get_allocation_win(get_mechanism):
    """
    Test allocation method of crowdsourcing contest with random tie-breaking rule
    """
    mechanism = get_mechanism(
        n_bidder=3, prices=[0.8, 0.2, 0.0], tie_breaking="win", type="value"
    )
    bids = np.array([[0, 1, 2, 0], [0, 1, 1, 1], [0, 0, 1, 2]])
    result = np.array(
        [
            [[1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
            [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]],
            [[1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
        ]
    )
    # test allocation
    for idx in range(3):
        allocation = mechanism.get_allocation(bids, idx)
        assert isinstance(allocation, np.ndarray), "allocation returns array"
        assert np.allclose(allocation, result[idx]), f"allocation agent {idx}"


def test_get_payoff(get_mechanism):
    """
    Test payoff method of crowdsourcing contest for agent 0
    """
    mechanism = get_mechanism(
        n_bidder=3, prices=[0.8, 0.2, 0.0], tie_breaking="random", type="value"
    )
    bids = np.array([[0, 1, 2, 0], [0, 1, 1, 1], [0, 0, 1, 2]])
    result = [
        (0.8 / 3 + 0.2 / 3 + 0.0 / 3) * 1 - 0,
        (0.8 / 2 + 0.2 / 2) * 0.5 - 1,
        0.8 * 0.0 - 2,
        0.0,
    ]

    valuation = np.array([1.0, 0.5, 0.0, 1.0])
    allocation = mechanism.get_allocation(bids, 0)
    payment = mechanism.get_payment(bids, allocation, 0)
    payoff = mechanism.get_payoff(allocation, valuation, payment)
    assert np.allclose(payoff, result), "payoff agent 0"

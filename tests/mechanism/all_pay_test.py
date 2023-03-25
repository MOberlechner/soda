"""
This module tests the all-pay mechanism
"""
import numpy as np
import pytest

from src.game import Game
from src.learner.gradient import Gradient
from src.mechanism.all_pay import AllPayAuction
from src.strategy import Strategy


@pytest.fixture
def get_mechanism():
    """
    Function to create mechanism
    """

    def _mechanism(
        n_bidder: int, utility_type: str = "RN", tie_breaking: str = "lose"
    ) -> AllPayAuction:
        """
        create mechanism
        """
        bidder = ["1"] * n_bidder
        o_space = {"1": [0.0, 1.0]}
        a_space = {"1": [0.0, 1.0]}
        param_prior = {"distribution": "uniform"}
        param_util = {
            "payment_rule": "first_price",
            "tie_breaking": tie_breaking,
            "type": "value",
            "utility_type": utility_type,
            "risk_parameter": 0.5,
        }
        return AllPayAuction(bidder, o_space, a_space, param_prior, param_util)

    return _mechanism


def test_create_mechamism(get_mechanism):
    """
    Test if all-pay mechanism can be created
    """
    mechanism = get_mechanism(3)
    assert isinstance(mechanism, AllPayAuction), "create all-pay auction mechanism"


def test_get_allocation(get_mechanism):
    """
    Test allocation method of single-item auction
    Similar to single-item auctions, zero-bids can not win

    """
    mechanism = get_mechanism(2)
    bids = np.array([[0, 1, 2, 3, 3, 0], [1, 1, 1, 3, 2, 0]])

    # test allocation with tie-breaking random
    mechanism.tie_breaking = "random"
    allocation = mechanism.get_allocation(bids, 0)
    assert np.allclose(
        allocation, [0.0, 0.5, 1.0, 0.5, 1.0, 0.5]
    ), "allocation with tie-breaking: random"

    # test alloaction with tie-breaking lose
    mechanism.tie_breaking = "lose"
    allocation = mechanism.get_allocation(bids, 0)
    assert np.allclose(
        allocation, [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    ), "allocation with tie-breaking: lose"


testdata = [
    ("RN", "random", [-0.5, -2 / 3, -2.0, -3.0, 0.0, 2 / 3, -0.375]),
    ("RN", "lose", [-1.0, -1.0, -2.0, -3.0, 0.0, 0.0, -0.375]),
    (
        "CRRA",
        "random",
        [
            -0.5,
            -2 / 3,
            -np.sqrt(2),
            -1 / 2 * np.sqrt(3) - 1 / 2 * np.sqrt(3),
            0.0,
            1 / 3 * np.sqrt(2),
            -np.sqrt(0.375),
        ],
    ),
    ("CRRA", "lose", [-1, -1, -np.sqrt(2), -np.sqrt(3), 0.0, 0, -np.sqrt(0.375)]),
]


@pytest.mark.parametrize("utility_type, tie_breaking, result", testdata)
def test_payoff(get_mechanism, utility_type, tie_breaking, result):
    """
    Test payoff method for all-pay auction
    """
    mechanism = get_mechanism(3, utility_type, tie_breaking)
    bids = np.array(
        [
            [0, 1, 3, 3, 1, 0, 0],
            [1, 1, 2, 3, 2, 0, 0.5],  # test this players payoff
            [1, 1, 1, 1, 1, 0, 0],
        ]
    )
    val = np.array([1, 1, 1, 0, 2, 2, 0.125])

    allocation = mechanism.get_allocation(bids, 1)
    payment = mechanism.get_payment(bids, allocation, 1)
    payoff = mechanism.get_payoff(allocation, val, payment)
    assert np.allclose(payoff, result)


def test_compute_expected_revenue(get_mechanism):
    """
    Test computation of expected revenue in all-pay auction (first-price)
    """
    mechanism = get_mechanism(3)
    bids = np.array([[0, 1, 2, 3, 1, 0], [1, 1, 1, 3, 2, 2], [1, 1, 1, 1, 1, 1]])
    revenue = mechanism.compute_expected_revenue(bids)
    assert np.isclose(revenue, 1 / 6 * (2 + 3 + 4 + 7 + 4 + 3))


testdata = [
    ("RN", "lose"),
    ("RN", "random"),
    ("CARA", "lose"),
    ("CARA", "random"),
    ("CRRA", "lose"),
    ("CRRA", "random"),
]


@pytest.mark.parametrize("utility_type, tie_breaking", testdata)
def test_own_gradient(get_mechanism, utility_type, tie_breaking):
    """
    Function to compare gradient computation of mechanism (fast) and gradient class (slow)
    """
    # setup setting
    mechanism = get_mechanism(3, utility_type, tie_breaking)
    assert mechanism.own_gradient, "check if setting has own_gradient method"

    # setup gradient
    mechanism.own_gradient = False
    game = Game(mechanism, 21, 19)
    game.get_utility()
    strategies = {"1": Strategy("1", game)}
    gradient = Gradient()
    gradient.prepare(game, strategies)

    for t in range(1):

        strategies["1"].initialize("random")
        util_gradient = gradient.compute(game, strategies, "1")
        own_gradient = mechanism.compute_gradient(game, strategies, "1")
        assert np.allclose(util_gradient, own_gradient), "equality gradient"

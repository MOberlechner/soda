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

    def _mechanism(n_bidder: int) -> AllPayAuction:
        """
        create mechanism
        """
        bidder = ["1"] * n_bidder
        o_space = {"1": [0.0, 1.0]}
        a_space = {"1": [0.0, 1.0]}
        param_prior = {"distribution": "uniform"}
        param_util = {
            "payment_rule": "first_price",
            "tie_breaking": "lose",
            "type": "valuation",
            "utility_type": "RN",
            "risk_parameter": 0.3,
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


def test_payoff_risk_neutral(get_mechanism):
    """
    Test payoff method for all-pay auction with risk-neutral agents
    """
    mechanism = get_mechanism(3)
    bids = np.array([[0, 1, 2, 3, 1, 0], [1, 1, 1, 3, 2, 2], [1, 1, 1, 1, 1, 1]])
    val = np.array([1, 1, 1, 0, 2, 0])

    # test allocation with tie-breaking random
    mechanism.tie_breaking = "random"
    allocation = mechanism.get_allocation(bids, 1)
    payment = mechanism.get_payment(bids, allocation, 1)
    payoff = mechanism.get_payoff(allocation, val, payment)
    assert np.allclose(payoff, [-0.5, -2 / 3, -1.0, -3.0, 0.0, -2.0])


def test_own_gradient(get_mechanism):
    """
    Function to compare gradient computation of mechanism (fast) and gradient class (slow)
    """
    for utility_type in ["RN", "CRRA"]:

        # setup setting
        mechanism = get_mechanism(2)
        mechanism.tie_breaking = "lose"
        mechanism.utility_type = utility_type
        mechanism.reserve_price = 0.1

        assert mechanism.own_gradient, "check if setting has own_gradient method"

        # setup gradient
        mechanism.own_gradient = False
        game = Game(mechanism, 21, 19)
        game.get_utility()
        strategies = {"1": Strategy("1", game)}
        gradient = Gradient()
        gradient.prepare(game, strategies)

        for t in range(5):

            strategies["1"].initialize("random")

            gradient.compute(game, strategies, "1")
            own_gradient = mechanism.compute_gradient(game, strategies, "1")

            assert np.allclose(
                gradient.x["1"], own_gradient
            ), f"equality gradient for {utility_type}"

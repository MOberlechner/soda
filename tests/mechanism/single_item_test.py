"""
This module tests the single-item mechanism
"""
import numpy as np
import pytest

from soda.game import Game
from soda.learner.gradient import Gradient
from soda.mechanism.single_item import SingleItemAuction
from soda.strategy import Strategy


@pytest.fixture
def get_mechanism():
    """
    Function to create mechanism
    """

    def _mechanism(n_bidder: int):
        bidder = ["1"] * n_bidder
        o_space = {i: [0, 1] for i in bidder}
        a_space = {i: [0, 1] for i in bidder}
        param_prior = {"distribution": "uniform"}
        param_util = {
            "payment_rule": "first_price",
            "tie_breaking": "random",
            "utility_type": "QL",
            "risk_parameter": 0.5,
            "budget": 1.01,
        }
        return SingleItemAuction(bidder, o_space, a_space, param_prior, param_util)

    return _mechanism


def test_create_mechamism(get_mechanism):
    """
    Test if single-item mechanism can be created
    """
    mechanism = get_mechanism(3)
    assert isinstance(
        mechanism, SingleItemAuction
    ), "create single-item auction mechanism"


def test_get_allocation(get_mechanism):
    """
    Test allocation method of single-item auction

    """
    mechanism = get_mechanism(2)

    # test allocation with tie-breaking random
    mechanism.tie_breaking = "random"
    bids = np.array([[0, 1, 2, 3, 3, 0], [1, 1, 1, 3, 2, 0]])
    allocation = mechanism.get_allocation(bids, 0)
    assert np.allclose(
        allocation, [0.0, 0.5, 1.0, 0.5, 1.0, 0.0]
    ), "allocation with tie-breaking: random"

    # test alloaction with tie-breaking lose
    mechanism.tie_breaking = "lose"
    bids = np.array([[0, 1, 2, 3, 3], [1, 1, 1, 3, 2]])
    allocation = mechanism.get_allocation(bids, 0)
    assert np.allclose(
        allocation, [0.0, 0.0, 1.0, 0.0, 1.0]
    ), "allocation with tie-breaking: lose"


def test_get_payment(get_mechanism):
    """
    Function to test payment method in single-item auction

    Note that second price p_i is defined by max b_j with j != i (analogous third price) (independent of b_i)
    """
    mechanism = get_mechanism(3)
    bids = np.array([[1, 2, 3, 4, 5], [2, 1, 3, 1, 0], [2, 2, 3, 2, 0]])
    allocation = mechanism.get_allocation(bids, 0)

    mechanism.payment_rule = "first_price"
    payments = mechanism.get_payment(bids, allocation, 0)
    assert np.allclose(payments, [0, 2, 3, 4, 5]), "first_price payment rule"

    mechanism.payment_rule = "second_price"
    payments = mechanism.get_payment(bids, allocation, 0)
    assert np.allclose(payments, [0, 2, 3, 2, 0]), "second_price payment rule"

    mechanism.payment_rule = "third_price"
    payments = mechanism.get_payment(bids, allocation, 0)
    assert np.allclose(payments, [0, 1, 3, 1, 0]), "third_price payment rule"


def test_get_payoff(get_mechanism):
    mechanism = get_mechanism(2)

    # test for simulations (i.e. number valuations = number bids)
    valuation_sim = np.array([1, 1, 1, 0, 2])
    # test for computation of utility array (i.e. number valuations != number bids)
    valuation_util = np.array([1, 0]).reshape(2, 1)

    allocation = np.array([1, 1, 0, 1, 1])
    payment = np.array([1, 2, 0, 0, 0.5])

    mechanism.utility_type = "QL"
    payoff = mechanism.get_payoff(allocation, valuation_sim, payment)
    assert np.allclose(
        payoff, np.array([0, -1, 0, 0, 1.5])
    ), "quasi-lineare (QL) utility-type, dim val = dim bids"
    payoff = mechanism.get_payoff(allocation, valuation_util, payment)
    assert np.allclose(
        payoff, np.array([[0, -1, 0, 1, 0.5], [-1, -2, 0, 0, -0.5]])
    ), "quasi-lineare utility-type, dim val != dim bids"

    mechanism.utility_type = "ROI"
    payoff = mechanism.get_payoff(allocation, valuation_sim, payment)
    assert np.allclose(
        payoff, np.array([0, -0.5, 0, 0, 3])
    ), "return-of-investment (ROI) utility type"

    mechanism.utility_type = "ROS"
    payoff = mechanism.get_payoff(allocation, valuation_sim, payment)
    assert np.allclose(
        payoff, np.array([1, 0.5, 0, 0, 4])
    ), "return-of-spent (ROS) utility type"


testdata = [
    ("QL", "lose"),
    ("QL", "random"),
    ("ROI", "lose"),
    ("ROI", "random"),
    ("ROS", "lose"),
    ("ROS", "random"),
    ("ROSB", "lose"),
    ("ROSB", "random"),
    ("CRRA", "lose"),
    ("CRRA", "random"),
]


@pytest.mark.parametrize("utility_type, tie_breaking", testdata)
def test_own_gradient(get_mechanism, utility_type, tie_breaking):
    """
    Function to compare gradient computation of mechanism (fast) and gradient class (slow)
    """
    # setup setting
    mechanism = get_mechanism(2)
    mechanism.tie_breaking = tie_breaking
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

        util_gradient = gradient.compute(game, strategies, "1")
        own_gradient = mechanism.compute_gradient(game, strategies, "1")

        assert np.allclose(
            util_gradient, own_gradient
        ), f"equality gradient, setting: {tie_breaking}, {utility_type}"

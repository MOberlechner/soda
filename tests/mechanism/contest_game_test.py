"""
This module tests the contest games
"""
import numpy as np
import pytest

from soda.game import Game
from soda.learner.gradient import Gradient
from soda.mechanism.contest_game import ContestGame
from soda.strategy import Strategy


@pytest.fixture
def get_mechanism():
    """
    Function to create mechanism
    """

    def _mechanism(
        n_bidder: int,
    ) -> ContestGame:
        """
        create mechanism
        """
        bidder = ["1"] * n_bidder
        o_space = {"1": [0.0, 1.0]}
        a_space = {"1": [0.0, 1.0]}
        param_prior = {"distribution": "uniform"}
        param_util = {
            "csf": "ratio_form",
            "csf_parameter": 1.0,
            "type": "value",
        }
        return ContestGame(bidder, o_space, a_space, param_prior, param_util)

    return _mechanism


def test_create_mechamism(get_mechanism):
    """
    Test if contest mechanism can be created
    """
    mechanism = get_mechanism(3)
    assert isinstance(mechanism, ContestGame), "create contest"


def test_get_allocation(get_mechanism):
    """
    Test allocation method of single-item auction
    Similar to single-item auctions, zero-bids can not win

    """
    mechanism = get_mechanism(2)
    bids = np.array([[0, 1, 2, 3, 3, 0], [0, 1, 1, 3, 2, 1]])

    # test allocation for ratio form (param=1)
    allocation = mechanism.get_allocation(bids, 0)
    assert np.allclose(
        allocation, [0.5, 0.5, 2 / 3, 0.5, 3 / 5, 0.0]
    ), "allocation with ratio-form: csf_parameter: 1"

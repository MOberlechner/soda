"""
This module tests the game class, which is used to create the approximation game from a given mechanism
"""
import numpy as np
import pytest

from soda.game import Game
from soda.mechanism.single_item import SingleItemAuction


@pytest.fixture
def get_mechanism():
    """
    Method to create game from FPSB mechanism
    """
    bidder = ["1"] * 2
    o_space = {i: [0, 1] for i in bidder}
    a_space = {i: [0, 1] for i in bidder}
    param_prior = {"distribution": "uniform"}
    param_util = {
        "payment_rule": "first_price",
        "tie_breaking": "random",
        "utility_type": "QL",
    }
    return SingleItemAuction(bidder, o_space, a_space, param_prior, param_util)


def test_create_game(get_mechanism):
    game = Game(get_mechanism, 7, 9)
    assert isinstance(game, Game), "create instance of Game"
    assert len(game.o_discr["1"]) == 7, "discretization points observation space"
    assert len(game.a_discr["1"]) == 9, "discretization points action space"


def test_discr_interval():
    """
    Test if discrete intervals are created correctly
    """
    # midpoint is True
    assert np.array_equal(
        Game.discr_interval(0, 1, 5, False), np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    )
    assert np.array_equal(Game.discr_interval(1, 1, 1, True), np.array([1.0]))

    # midpoint is False
    assert np.array_equal(Game.discr_interval(0, 1, 2, True), np.array([0.25, 0.75]))
    assert np.array_equal(Game.discr_interval(0, 1, 2, False), np.array([0.0, 1.0]))


def test_discr_spaces():
    """
    Test if discretized spaces are created correctly
    """
    # 1 dim space
    interval_1d = [0, 1]
    assert np.array_equal(
        Game.discr_spaces(interval_1d, 3, False), np.array([0, 0.5, 1])
    )
    # 2 dim space
    interval_2d = [[0, 1], [1, 2]]
    assert np.array_equal(
        Game.discr_spaces(interval_2d, 2, True), np.array([[0.25, 0.75], [1.25, 1.75]])
    )

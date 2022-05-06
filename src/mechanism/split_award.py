from typing import Dict, List

import numpy as np

from .mechanism import Mechanism

# -------------------------------------------------------------------------------------------------------------------- #
#                                          COMBINATORAL AUCTION - LLG                                                  #
# -------------------------------------------------------------------------------------------------------------------- #


class SplitAwardAuction(Mechanism):
    """Split-Award Auction

    The split award auction is a procurement auction with two bidders.

    Attributes:
        split (float)
        scale (float)

    """

    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        param_prior: Dict[str, str],
        param_util: Dict,
    ):
        super().__init__(bidder, o_space, a_space, param_prior, param_util)

        self.name = "split_award"
        self.split = 0.5
        self.scale = param_util["scale"] if "split" in param_util else 0.3

    def utility(self, obs: np.ndarray, bids: np.ndarray, idx: int):
        """Compute Utility for Split Award Auction

        Args:
            obs (np.ndarray): observations/valuations from each agent
            bids (np.ndarray): bids (2 dim: single,split) from each agent
            idx (int): index of agent

        """

        # test input
        if bids.shape[0] != self.n_bidder:
            raise ValueError("wrong format of bids")
        elif idx >= self.n_bidder:
            raise ValueError("bidder with index {idx} not avaible".format(idx))

        # if True: we want each outcome for every observation,  each outcome belongs to one observation
        if obs.shape != bids[idx].shape:
            obs = obs.reshape(len(obs), 1)

        idx_single, idx_split = 0, 1
        bids_single = np.array([bids[i][idx_single] for i in range(self.n_bidder)])
        bids_split = np.array([bids[i][idx_split] for i in range(self.n_bidder)])

        # determine allocation
        win_split = np.where(bids_split.sum(axis=0) <= bids_single.min(axis=0), 1, 0)

        if self.param_util["tie_breaking"] == "random":
            win_single = np.where(
                bids_single[idx] <= np.delete(bids_single, idx, 0).min(axis=0), 1, 0
            )
            num_winner_single = (bids_single[idx] == bids_single).sum(axis=0)

        elif self.param_util["tie_breaking"] == "lose":
            win_single = np.where(
                bids_single[idx] < np.delete(bids_single, idx, 0).min(axis=0), 1, 0
            )
            num_winner_single = np.ones(win_single.shape)
        else:
            raise NotImplementedError(
                'tie-breaking rule "{}" not implemented'.format(
                    self.param_util["tie_breaking"]
                )
            )

        return win_split * (bids_split[idx] - self.scale * obs) + (
            1 - win_split
        ) * win_single * 1 / num_winner_single * (bids_single[idx] - obs)

    def get_bne(self, agent: str, obs: np.ndarray):
        pass

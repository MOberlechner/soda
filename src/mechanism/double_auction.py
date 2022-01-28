import numpy as np
from typing import List, Dict


def util(val: np.ndarray, bids: np.ndarray, bidder: List[str], idx: int, param: Dict = {}):
    """

    Parameters
    ----------
    val : np.ndarray, one-dimensional valuation
    bids : np.ndarray,
    bidder : list
    idx : int
    param : dict

    Returns
    -------

    """

    # deterimine parameter
    n_bidder, n_profiles = bids.shape
    n_seller = bidder.count('S')
    n_buyer = bidder.count('B')

    # test input
    if bids.shape[0] != len(bidder):
        raise ValueError('wrong format of bids')
    if n_seller + n_buyer != n_bidder:
        raise ValueError('bidder must only contain buyers (B) or sellers (S)')

    # risk parameter & payment rule
    risk = param['risk'] if 'risk' in param.keys() else 1
    payment_rule = param['payment_rule'] if 'payment_rule' in param.keys() else 'average'

    # determine where trade takes place
    seller = np.array(bidder) == 'S'
    buyer = np.array(bidder) == 'B'
    number_trades = (-np.sort(-bids[buyer], axis=0) >= np.sort(bids[seller], axis=0)).sum(axis=0)

    trade_bool = np.argsort(bids[np.array(bidder) == bidder[idx]], axis=0)[idx] < number_trades

    if payment_rule == 'average':
        payment = 0.5*(-np.sort(-bids[buyer], axis=0)[number_trades-1, np.arange(n_profiles)] +
                       np.sort(bids[seller], axis=0)[number_trades-1, np.arange(n_profiles)])

    elif payment_rule == 'vcg':
        print('VCG not yet implemented')

    # if True: we want each outcome for every valuation,  each outcome belongs to one valuation
    if val.shape != bids[idx].shape:
        val = val.reshape(len(val),1)

    return trade_bool * (1 if bidder[idx] == 'B' else -1) * np.sign(val-payment) * np.abs(val - payment)**risk




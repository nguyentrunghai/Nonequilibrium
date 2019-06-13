"""
plot rmse of pmf data from Van A Ngo from Los Alamos
"""
from __future__ import print_function
from __future__ import division

import numpy as np

def _index_closest_to(ref_array, query_number):
    """
    find the index of an element of ref_array that is closest to query_number
    :param ref_array: 1D array
    :param query_number: float
    :return: idx, int
    """
    assert ref_array.ndim == 1, "ref_array must be 1D"
    abs_diff = np.abs(ref_array - query_number)
    idx = abs_diff.argmin()
    return idx

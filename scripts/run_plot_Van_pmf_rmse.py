"""
plot rmse of pmf data from Van A Ngo from Los Alamos
"""
from __future__ import print_function
from __future__ import division

import numpy as np

from _fe_pmf_plot_utils import min_to_zero


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


def _indices_closest_to(ref_array, query_array):
    """
    find a list of indices of elements in ref_array that are each closest to each element in query_array
    :param ref_array: 1D array
    :param query_array: 1D array
    :return: indices, list
    """
    indices = [_index_closest_to(ref_array, test) for test in query_array]
    return indices


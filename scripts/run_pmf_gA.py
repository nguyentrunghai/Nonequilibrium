"""
calculate pmf for gA
"""

from __future__ import print_function

import argparse

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--work_data_file", type=str, default="work.nc")

args = parser.parse_args()


def _pmf_bin_edges(lower, upper, nbins, symmetric_center):

    if symmetric_center is not None:
        assert lower < symmetric_center < upper, "symmetric_center is not in between [min, max]"

        left_interval = symmetric_center - lower
        right_interval = upper - symmetric_center

        interval = np.max([left_interval, right_interval])

        lower = symmetric_center - interval
        upper = symmetric_center + interval

    bin_edges = np.linspace(lower, upper, nbins + 1)

    return bin_edges
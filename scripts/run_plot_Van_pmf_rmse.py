"""
plot rmse of pmf data from Van A Ngo from Los Alamos
"""
from __future__ import print_function
from __future__ import division

import os
import argparse

import numpy as np

from _fe_pmf_plot_utils import min_to_zero
from _plots import plot_lines


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


def _absolute_deviation(ref_pmf, pmf):
    """
    :param ref_pmf, 2D array of shape (2, n1_bins)
    :param pmf, 2D array of shape (2, n2_bins)
    :return: abs_dev, 2D array of shape (2, n2_bins)
    """
    align_indices = _indices_closest_to(ref_pmf[:, 0], pmf[:, 0])
    bin_centers = pmf[:, 0]

    ref = min_to_zero(ref_pmf[align_indices, 1])
    target = min_to_zero(pmf[:, 1])

    ad = np.abs(target - ref)
    abs_dev = np.hstack((bin_centers[:, np.newaxis], ad[:, np.newaxis]))
    return abs_dev


def _down_sampling(array, n_points):
    indices = np.linspace(0, array.shape[0] - 1, n_points)
    indices = np.round(indices)
    indices = indices.astype(int)
    return indices


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./")
parser.add_argument("--data_files", type=str,
                    default="unidirectional.dat bidirectional.dat s1.dat MA+WHAM.dat us_pmf.dat")

parser.add_argument("--labels", type=str, default="u b s b+WHAM us")
parser.add_argument("--colors", type=str, default="blue green red cyan black")
parser.add_argument("--markers", type=str, default="< > ^ v .")

parser.add_argument("--units", type=str, default="Rt Rt Rt kcal_per_mol kcal_per_mol")

parser.add_argument("--legend_ncol", type=int, default=2)

parser.add_argument("--n_points_to_plot", type=int, default=40)

parser.add_argument("--xlabel", type=str, default="$z$ (nm)")
parser.add_argument("--ylabel", type=str, default="$|\Phi(z) - \Phi_{us}(z)|$ (RT)")

parser.add_argument("--xlimits", type=str, default="None")
parser.add_argument("--ylimits", type=str, default="None")

parser.add_argument("--out", type=str, default="abs_dev_pmf.pdf")

args = parser.parse_args()

KB = 0.0019872041   # kcal/mol/K
TEMPERATURE = 300.
BETA = 1/KB/TEMPERATURE

labels = args.labels.split()
print("labels:", labels)

data_files = [os.path.join(args.data_dir, f) for f in args.data_files.split()]
data_files = {label: data_file for label, data_file in zip(data_files, labels)}
print("data_files:", data_files)

colors = args.colors.split()
print("colors:", colors)

markers = args.markers.split()
print("markers:", markers)

units = args.units.split()
print("units:", units)

pmfs = {}
for label, unit in zip(labels, units):
    data = np.loadtxt(data_files[label])[:, 0:2]

    if unit == "kcal_per_mol":
        data[:, 1] *= BETA       # kcal/mol to KT/mol or RT

    pmfs[label] = data

labels_without_us = [label for label in labels if label != "us"]
abs_devs = {}
for label in labels_without_us:
    ad = _absolute_deviation(pmfs["us"], pmfs[label])
    abs_devs[label] = ad

xs = []
ys = []

for label in labels_without_us:
    x = abs_devs[label][:, 0]
    y = abs_devs[label][:, 1]

    x /= 10.    # angstrom to nm

    indices = _down_sampling(y, args.n_points_to_plot)
    y = y[indices]
    x = x[indices]

    xs.append(x)
    ys.append(y)

if args.xlimits.lower() != "none":
    xlimits = [float(s) for s in args.xlimits.split()]
else:
    xlimits = None

if args.ylimits.lower() != "none":
    ylimits = [float(s) for s in args.ylimits.split()]
else:
    ylimits = None


plot_lines(xs, ys,
           xlabel=args.xlabel, ylabel=args.ylabel,
           out=args.out,
           legends=labels_without_us,
           legend_pos="best",
           legend_ncol=args.legend_ncol,
           legend_fontsize=8,
           markers=markers,
           colors=colors,
           xlimits=xlimits,
           ylimits=ylimits,
           lw=1.0,
           markersize=3,
           alpha=1.,
           n_xtics=8,
           n_ytics=8)



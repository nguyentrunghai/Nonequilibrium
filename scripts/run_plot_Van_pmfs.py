"""
plot pmf data from Van A Ngo from Los Alamos
"""

from __future__ import print_function
from __future__ import division

import os
import argparse

import numpy as np

from _fe_pmf_plot_utils import min_to_zero
from _plots import plot_lines

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./")
parser.add_argument("--data_files", type=str,
                    default="unidirectional.dat bidirectional.dat s1.dat MA+WHAM.dat us_pmf.dat")

parser.add_argument("--labels", type=str, default="u b s b+WHAM us")
parser.add_argument("--colors", type=str, default="blue green red cyan black")
parser.add_argument("--markers", type=str, default="< > ^ v .")

parser.add_argument("--units", type=str, default="Rt Rt Rt kcal_per_mol kcal_per_mol")

parser.add_argument("--legend_ncol", type=int, default=2)

parser.add_argument("--which_to_down_sample", type=str, default="n n n n n")
parser.add_argument("--n_points_to_plot", type=int, default=40)

parser.add_argument("--xlabel", type=str, default="$z$ (nm)")
parser.add_argument("--ylabel", type=str, default="$\Phi(z)$ (RT)")

parser.add_argument("--xlimits", type=str, default="None")
parser.add_argument("--ylimits", type=str, default="None")

parser.add_argument("--out", type=str, default="pmf.pdf")

args = parser.parse_args()

KB = 0.0019872041   # kcal/mol/K
TEMPERATURE = 300.
BETA = 1/KB/TEMPERATURE


def _down_sampling(array, n_points):
    indices = np.linspace(0, array.shape[0] - 1, n_points)
    indices = np.round(indices)
    indices = indices.astype(int)
    return indices


data_files = [os.path.join(args.data_dir, f) for f in args.data_files.split()]
print("data_files:", data_files)

labels = args.labels.split()
print("labels:", labels)

colors = args.colors.split()
print("colors:", colors)

markers = args.markers.split()
print("markers:", markers)

units = args.units.split()
print("units:", units)

which_to_down_sample = args.which_to_down_sample.split()
for flag in which_to_down_sample:
    if flag not in ["n", "y"]:
        raise ValueError("Unknown down-sampling flag")
print("which_to_down_sample:", which_to_down_sample)

xs = []
ys = []

for f, unit, ds_flag in zip(data_files, units, which_to_down_sample):
    data = np.loadtxt(f)
    x = data[:, 0] / 10.     # angstrom to nm
    y = data[:, 1]

    if unit == "kcal_per_mol":
        y *= BETA       # kcal/mol to KT/mol or RT

    if ds_flag == "y":
        indices = _down_sampling(y, args.n_points_to_plot)
        y = y[indices]
        x = x[indices]

    y = min_to_zero(y)

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
           legends=labels,
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

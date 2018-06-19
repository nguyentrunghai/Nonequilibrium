"""

"""

from __future__ import print_function

import os
import pickle
import argparse

import numpy as np

from _plots import plot_lines
from utils import bin_centers

parser = argparse.ArgumentParser()

parser.add_argument("--pull_pmf_dir", type=str, default="pull_fe_pmf")
parser.add_argument("--pull_estimators", type=str, default="u b s1 s2")

parser.add_argument("--us_pmf_file", type=str,default="pmf.dat")

parser.add_argument("--xlimits", type=str, default="None")
parser.add_argument("--ylimits_pmf", type=str, default="None")
parser.add_argument("--ylimits_rmse", type=str, default="None")

parser.add_argument("--xlabel", type=str, default="$d$ (nm)")
parser.add_argument("--ylabel_pmf", type=str, default="$\Phi (d)$ (kcal/mol)")
parser.add_argument("--ylabel_rmse", type=str, default="RMSE[$\Phi (d)$] (kcal/mol)")

parser.add_argument("--pmf_out", type=str, default="pmf.pdf")
parser.add_argument("--rmse_out", type=str, default="pmf_rmse.pdf")


args = parser.parse_args()


KB = 0.0019872041   # kcal/mol/K
TEMPERATURE = 300.

PMF_FILE_PREFIX = "pmf_"

MARKERS = ["<", ">", "^", "v", "o"]


def _shift_min_to_zero(array):
    return array - array.min()


def _load_wham_pmf(file_name):
    data = np.loadtxt(file_name)
    x = data[:, 0]
    y = data[:, 1]
    yerr = data[:, 2]
    return x, y, yerr


def _rmsd(list_of_data, ref_data):
    ref_data = _shift_min_to_zero(ref_data)

    for i in range(len(list_of_data)):
        assert list_of_data[i].shape == ref_data.shape, "data and ref_data do not have the same shape"
        list_of_data[i] = _shift_min_to_zero(list_of_data[i])

    err = ( np.array(list_of_data) - ref_data[None, :] )**2
    err = np.sqrt( err.mean(axis=0) )

    return err


pull_estimators = args.pull_estimators.split()
print("pull_estimators", pull_estimators)
pmfs_files = {est : os.path.join(args.pull_pmf_dir, PMF_FILE_PREFIX+est+".pkl")
             for est in pull_estimators}

print("pmfs_files", pmfs_files)
pull_pmfs = { est : pickle.load( open(pmfs_files[est], "r") ) for est in pull_estimators}

xs = [bin_centers(pull_pmfs[est]["pmf_bin_edges"]) for est in pull_estimators]
ys = [_shift_min_to_zero(pull_pmfs[est]["mean"]) for est in pull_estimators]
yerrs = [pull_pmfs[est]["std"] for est in pull_estimators]

ys = [y*KB*TEMPERATURE for y in ys]  # kT to kcal/mol
yerrs = [yerr*KB*TEMPERATURE for yerr in yerrs] # kT to kcal/mol


x, y, yerr = _load_wham_pmf(args.us_pmf_file)
x /= 10   # A to nm
y = _shift_min_to_zero(y)

xs.append(x)
ys.append(y)
yerrs.append(yerr)
us_pmf = {"bin_centers":x, "pmf":y, "std":yerr}

yerrs = [yerr/2. for yerr in yerrs]              # errors to be one std

if args.xlimits.lower() != "none":
    xlimits = [float(s) for s in args.xlimits.split()]
else:
    xlimits = None
print("xlimits = ", xlimits)

if args.ylimits_pmf.lower() != "none":
    ylimits_pmf = [float(s) for s in args.ylimits_pmf.split()]
else:
    ylimits_pmf = None
print("ylimits_pmf = ", ylimits_pmf)


plot_lines(xs, ys, yerrs=yerrs,
           xlabel=args.xlabel, ylabel=args.ylabel_pmf,
           out=args.pmf_out,
           legends=pull_estimators + ["us"],
           legend_pos="best",
           legend_fontsize=8,
           markers=MARKERS,
           xlimits=xlimits,
           ylimits=ylimits_pmf,
           lw=1.0,
           markersize=5,
           alpha=0.5,
           n_xtics=8,
           n_ytics=8)


# rmsd with respect to US
xs = []
ys = []

for est in pull_estimators:
    list_of_pmfs = pull_pmfs[est]["pmfs"].values()

    list_of_pmfs = [pmf*KB*TEMPERATURE for pmf in list_of_pmfs]

    x = bin_centers(pull_pmfs[est]["pmf_bin_edges"])
    xs.append(x)

    rmsd = _rmsd(list_of_pmfs, us_pmf["pmf"])
    ys.append(rmsd)

if args.ylimits_rmse.lower() != "none":
    ylimits_rmse = [float(s) for s in args.ylimits_rmse.split()]
else:
    ylimits_rmse = None
print("ylimits_rmse = ", ylimits_rmse)

plot_lines(xs, ys,
           xlabel=args.xlabel, ylabel=args.ylabel_rmse,
           out=args.rmse_out,
           legends=pull_estimators,
           legend_pos="best",
           legend_fontsize=8,
           markers=MARKERS,
           xlimits=xlimits,
           ylimits=ylimits_rmse,
           lw=1.0,
           markersize=5,
           alpha=0.5,
           n_xtics=8,
           n_ytics=8)

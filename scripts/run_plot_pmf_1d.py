"""
"""

from __future__ import print_function

import argparse
import pickle
import os

import numpy as np

from _plots import plot_lines
from utils import bin_centers

parser = argparse.ArgumentParser()

parser.add_argument("--pmf_dir", type=str,
                    default="fe_pmf")

parser.add_argument("--estimators", type=str, default="u b s1 s2")

parser.add_argument("--exact_pmf_file", type=str,
                    default="/home/tnguye46/nonequilibrium/1d/sym/fe_pmf/1000repeats/pmf_exact.pkl")

parser.add_argument("--xlimits", type=str, default="None")
parser.add_argument("--ylimits_pmf", type=str, default="None")
parser.add_argument("--ylimits_rmse", type=str, default="None")

parser.add_argument("--xlabel", type=str, default="$z$")
parser.add_argument("--ylabel_pmf", type=str, default="$\Phi (z)$")
parser.add_argument("--ylabel_rmse", type=str, default="RMSE[$\Phi (z)$]")

parser.add_argument("--pmf_out", type=str, default="pmf.pdf")
parser.add_argument("--rmse_out", type=str, default="pmf_rmse.pdf")

args = parser.parse_args()

PMF_FILE_PREFIX = "pmf_"

MARKERS = ["<", ">", "^", "v", "o"]


def _match_min(to_be_transformed, target):
    """
    transfrom such that to_be_transformed[argmin] == target[argmin]
    where argmin = np.argmin(target)
    """
    assert to_be_transformed.ndim == target.ndim == 1, "to_be_transformed and target must be 1d"
    assert to_be_transformed.shape == target.shape, "pmf_to_be_transformed and pmf_target must have the same shape"

    argmin = np.argmin(target)
    d = target[argmin] - to_be_transformed[argmin]
    transformed = to_be_transformed + d

    return transformed


def _rmse(list_of_est_val, true_val):
    """
    """
    assert true_val.ndim == 1, "true_val must be a 1d array"
    assert isinstance(list_of_est_val, list), "list_of_est_val must be a list"
    for est_val in list_of_est_val:
        assert est_val.shape == true_val.shape, "one of est_val has different shape than true_val"

    for i in range(len(list_of_est_val)):
        list_of_est_val[i] = _match_min(list_of_est_val[i], true_val)

    list_of_est_val = np.array(list_of_est_val)

    rmse = list_of_est_val - true_val[np.newaxis, :]
    rmse = rmse * rmse
    rmse = rmse.mean(axis=0)
    rmse = np.sqrt(rmse)

    return rmse


exact_pmf = pickle.load(open(args.exact_pmf_file, "r"))
pmf_bin_edges = exact_pmf["pmf_bin_edges"]
exact_pmf = exact_pmf["pmfs"]

estimators = args.estimators.split()
est_pmf_files = {est : os.path.join(args.pmf_dir, PMF_FILE_PREFIX + est + ".pkl") for est in estimators}
for f in est_pmf_files.values():
    assert os.path.exists(f), f + " does not exist."

est_pmfs = {est : pickle.load( open(est_pmf_files[est]) ) for est in estimators}

for est in estimators:
    est_pmfs[est]["pmfs"] = est_pmfs[est]["pmfs"].values()
    est_pmfs[est]["mean"] = _match_min(est_pmfs[est]["mean"], exact_pmf)

# plot pmf
xs = [bin_centers(est_pmfs[est]["pmf_bin_edges"][1:-1]) for est in estimators]
xs.append(bin_centers(pmf_bin_edges[1:-1]))

ys = [est_pmfs[est]["mean"][1:-1] for est in estimators]
ys.append(exact_pmf[1:-1])

yerrs = [est_pmfs[est]["std"][1:-1] for est in estimators]
yerrs = [yerr/2. for yerr in yerrs] # 1 std
yerrs.append(None)

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
           legends=estimators + ["exact"],
           markers=MARKERS,
           legend_pos="best",
           legend_fontsize=8,
           xlimits=xlimits,
           ylimits=ylimits_pmf,
           lw=1.0,
           markersize=5,
           alpha=0.5,
           n_xtics=8,
           n_ytics=8)



# plot RMSE
xs = [bin_centers(est_pmfs[est]["pmf_bin_edges"][1:-1]) for est in estimators]

ys = [_rmse(est_pmfs[est]["pmfs"], exact_pmf) for est in estimators]
ys = [y[1:-1] for y in ys]

if args.ylimits_rmse.lower() != "none":
    ylimits_rmse = [float(s) for s in args.ylimits_rmse.split()]
else:
    ylimits_rmse = None
print("ylimits_rmse = ", ylimits_rmse)

plot_lines(xs, ys,
           xlabel=args.xlabel, ylabel=args.ylabel_rmse,
           out=args.rmse_out,
           legends=estimators,
           markers=MARKERS,
           legend_pos="best",
           legend_fontsize=8,
           xlimits=xlimits,
           ylimits=ylimits_rmse,
           lw=1.0,
           markersize=5,
           alpha=0.5,
           n_xtics=8,
           n_ytics=8)


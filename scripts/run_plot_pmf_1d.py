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

parser.add_argument("--data_files", type=str,
                    default="../m1.5_to_p1.5_backto_m1.5/pmf_uf.pkl  ../m1.5_to_p1.5_backto_m1.5/pmf_b.pkl ../m1.5_to_p1.5_backto_m1.5/pmf_s1.pkl "\
                    + "../m1.5_to_p1.5/pmf_uf.pkl ../m1.5_to_p1.5/pmf_ur.pkl ../m1.5_to_p1.5/pmf_b.pkl" )

parser.add_argument("--data_estimator_pairs", type=str, default="s-u s-b s-s f-u r-u fr-b")

parser.add_argument("--reference_file_sym", type=str, default="../m1.5_to_p1.5_backto_m1.5/pmf_exact.pkl")
parser.add_argument("--reference_file_asym", type=str, default="../m1.5_to_p1.5/pmf_exact.pkl")

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

MARKERS = ["<", ">", "^", "v", "s", "d", None]


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


def _shift_min_to_zero(array):
    return array - array.min()


def _rmse(list_of_est_val, true_val):
    """
    """
    assert true_val.ndim == 1, "true_val must be a 1d array"
    assert isinstance(list_of_est_val, list), "list_of_est_val must be a list"
    for est_val in list_of_est_val:
        assert est_val.shape == true_val.shape, "one of est_val has different shape than true_val"

    true_val = _shift_min_to_zero(true_val)
    for i in range(len(list_of_est_val)):
        list_of_est_val[i] = _shift_min_to_zero(list_of_est_val[i])

    list_of_est_val = np.array(list_of_est_val)

    rmse = list_of_est_val - true_val[np.newaxis, :]
    rmse = rmse * rmse
    rmse = rmse.mean(axis=0)
    rmse = np.sqrt(rmse)

    return rmse


ref_pmf_sym = pickle.load(open(args.reference_file_sym, "r"))
bin_edges_sym = ref_pmf_sym["pmf_bin_edges"]
ref_pmf_sym = ref_pmf_sym["pmfs"]

ref_pmf_asym = pickle.load(open(args.reference_file_asym, "r"))
bin_edges_asym = ref_pmf_asym["pmf_bin_edges"]
ref_pmf_asym = ref_pmf_asym["pmfs"]

data_files = args.data_files.split()
for f in data_files:
    assert os.path.exists(f), f + " does not exist."

assert os.path.exists(args.reference_file_sym), args.reference_file_sym + " does not exist."
assert os.path.exists(args.reference_file_asym), args.reference_file_asym + " does not exist."


data_estimator_pairs = args.data_estimator_pairs.split()
for data_estimator_pair, data_file in zip(data_estimator_pairs, data_files):
    print(data_estimator_pair, ": ", data_file)
print("reference_file_sym: " + args.reference_file_sym)
print("reference_file_asym: " + args.reference_file_asym)

assert len(data_files) == len(data_estimator_pairs), "data_files and data_estimator_pairs must have the same len"

loaded_data = [pickle.load(open(f, "r")) for f in data_files]

# plot pmf
xs = []
ys = []
yerrs = []
for data, data_estimator_pair in zip(loaded_data, data_estimator_pairs):
    xs.append( bin_centers(data["pmf_bin_edges"]) )
    ys.append( _shift_min_to_zero(data["mean"]) )
    yerrs.append(data["std"] / 2. )

xs.append( bin_centers(bin_edges_sym) )
ys.append( _shift_min_to_zero(ref_pmf_sym) )
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

legends = data_estimator_pairs + ["exact"]
plot_lines(xs, ys, yerrs=yerrs,
           xlabel=args.xlabel, ylabel=args.ylabel_pmf,
           out=args.pmf_out,
           legends=legends,
           markers=MARKERS,
           legend_pos="best",
           legend_fontsize=7,
           xlimits=xlimits,
           ylimits=ylimits_pmf,
           lw=1.0,
           markersize=4,
           alpha=1.,
           n_xtics=8,
           n_ytics=8)


# plot RMSE
xs = []
ys = []
for data, data_estimator_pair in zip(loaded_data, data_estimator_pairs):
    xs.append(bin_centers(data["pmf_bin_edges"]))
    pmfs = data["pmfs"].values()

    if data_estimator_pair.split("-")[0] == "s":
        y = _rmse(pmfs, ref_pmf_sym)
    else:
        y = _rmse(pmfs, ref_pmf_asym)
    ys.append(y)

if args.ylimits_rmse.lower() != "none":
    ylimits_rmse = [float(s) for s in args.ylimits_rmse.split()]
else:
    ylimits_rmse = None
print("ylimits_rmse = ", ylimits_rmse)

legends = data_estimator_pairs
plot_lines(xs, ys,
           xlabel=args.xlabel, ylabel=args.ylabel_rmse,
           out=args.rmse_out,
           legends=legends,
           markers=MARKERS,
           legend_pos="best",
           legend_fontsize=7,
           xlimits=xlimits,
           ylimits=ylimits_rmse,
           lw=1.0,
           markersize=4,
           alpha=1.,
           n_xtics=8,
           n_ytics=8)


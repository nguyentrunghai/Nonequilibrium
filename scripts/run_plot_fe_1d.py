"""
"""

from __future__ import print_function

import argparse
import pickle
import os

import numpy as np

from _plots import plot_lines

parser = argparse.ArgumentParser()

parser.add_argument("--data_files", type=str, default="file1.pkl file2.pkl" )

parser.add_argument("--data_estimator_pairs", type=str, default="s-u s-b s-s f-u r-u fr-b")

parser.add_argument("--reference_file_sym", type=str, default="fe_numerical.pkl")
parser.add_argument("--reference_file_asym", type=str, default="fe_numerical.pkl")

parser.add_argument("--nsamples_sym", type=int, default=19)
parser.add_argument("--nsamples_asym", type=int, default=10)

parser.add_argument("--xlimits", type=str, default="None")
parser.add_argument("--ylimits_fe", type=str, default="None")
parser.add_argument("--ylimits_rmse", type=str, default="None")

parser.add_argument("--xlabel", type=str, default="$\lambda$")
parser.add_argument("--ylabel_fe", type=str, default="$\Delta F_{\lambda}$")
parser.add_argument("--ylabel_rmse", type=str, default="RMSE[$\Delta F_{\lambda}$]")

parser.add_argument("--fe_out", type=str, default="fe.pdf")
parser.add_argument("--rmse_out", type=str, default="fe_rmse.pdf")

args = parser.parse_args()


MARKERS = ["<", ">", "^", "v", "s", "d", "."]


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


def equal_stride_subsample(data, nsamples):
    assert nsamples < len(data), "nsamples must less than len of data"
    new_index = np.linspace(0, len(data)-1, nsamples, dtype=int)
    return data[new_index]


ref_fe_sym = pickle.load( open(args.reference_file_sym, "r") )
lambda_sym = ref_fe_sym["lambdas"]
ref_fe_sym = ref_fe_sym["fes"]

ref_fe_asym = pickle.load( open(args.reference_file_asym, "r") )
lambda_asym = ref_fe_asym["lambdas"]
ref_fe_asym = ref_fe_asym["fes"]


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

# plot fe
xs = []
ys = []
yerrs = []
for data, data_estimator_pair in zip(loaded_data, data_estimator_pairs):
    if data_estimator_pair.split("-")[0] == "s":
        x = equal_stride_subsample(data["lambdas"], args.nsamples_sym)
        y = equal_stride_subsample(data["mean"], args.nsamples_sym)
        yerr = equal_stride_subsample(data["std"], args.nsamples_sym)
    else:
        x = equal_stride_subsample(data["lambdas"], args.nsamples_asym)
        y = equal_stride_subsample(data["mean"], args.nsamples_asym)
        yerr = equal_stride_subsample(data["std"], args.nsamples_asym)

    if data_estimator_pair == "r-u":
        shift = ref_fe_asym[-1] - y[0]
        y += shift

    xs.append(x)
    ys.append(y)
    yerrs.append(yerr/2)  # error bars are one std


xs.append( equal_stride_subsample(lambda_sym, args.nsamples_sym) )
ys.append( equal_stride_subsample(ref_fe_sym, args.nsamples_sym) )
yerrs.append(None)

if args.xlimits.lower() != "none":
    xlimits = [float(s) for s in args.xlimits.split()]
else:
    xlimits = None
print("xlimits = ", xlimits)

if args.ylimits_fe.lower() != "none":
    ylimits_fe = [float(s) for s in args.ylimits_fe.split()]
else:
    ylimits_fe = None
print("ylimits_fe = ", ylimits_fe)

legends = data_estimator_pairs + ["num"]
plot_lines(xs, ys, yerrs=yerrs,
           xlabel=args.xlabel, ylabel=args.ylabel_fe,
           out=args.fe_out,
           legends=legends,
           markers=MARKERS,
           legend_pos="best",
           legend_fontsize=7,
           xlimits=xlimits,
           ylimits=ylimits_fe,
           lw=1.0,
           markersize=4,
           alpha=1.,
           n_xtics=8,
           n_ytics=8)


# plot RMSE
xs = []
ys = []
for data, data_estimator_pair in zip(loaded_data, data_estimator_pairs):
    fes = data["fes"].values()

    if data_estimator_pair == "r-u":
        for i in range(len(fes)):
            fes[i] = fes[i][::-1]

    if data_estimator_pair.split("-")[0] == "s":
        x = equal_stride_subsample(data["lambdas"], args.nsamples_sym)
        y = _rmse(fes, ref_fe_sym)
        y = equal_stride_subsample(y, args.nsamples_sym)
    else:
        x = equal_stride_subsample(data["lambdas"], args.nsamples_asym)
        y = _rmse(fes, ref_fe_asym)
        y = equal_stride_subsample(y, args.nsamples_asym)

    xs.append(x)
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


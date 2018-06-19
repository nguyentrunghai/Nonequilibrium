"""
"""

from __future__ import print_function

import argparse
import pickle
import os

import numpy as np

from _plots import plot_lines

parser = argparse.ArgumentParser()

parser.add_argument("--fe_dir", type=str,
                    default="fe_pmf")

parser.add_argument("--estimators", type=str, default="u b s1 s2")

parser.add_argument("--num_fe_file", type=str,
                    default="fe_numerical.pkl")

parser.add_argument("--nsamples", type=int, default=20)

parser.add_argument("--xlimits", type=str, default="None")
parser.add_argument("--ylimits_fe", type=str, default="None")
parser.add_argument("--ylimits_rmse", type=str, default="None")

parser.add_argument("--xlabel", type=str, default="$t$")
parser.add_argument("--ylabel_fe", type=str, default="$\Delta F_t$")
parser.add_argument("--ylabel_rmse", type=str, default="RMSE[$\Delta F_t$]")

parser.add_argument("--fe_out", type=str, default="fe.pdf")
parser.add_argument("--rmse_out", type=str, default="fe_rmse.pdf")

args = parser.parse_args()

FE_FILE_PREFIX = "fe_"

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


def equal_stride_subsample(data, nsamples):
    assert nsamples < len(data), "nsamples must less than len of data"
    new_index = np.linspace(0, len(data)-1, nsamples, dtype=int)
    return data[new_index]


num_fe = pickle.load(open(args.num_fe_file, "r"))
pulling_times = num_fe["pulling_times"]
num_fe = num_fe["fes"]

estimators = args.estimators.split()
est_fe_files = {est : os.path.join(args.fe_dir, FE_FILE_PREFIX + est + ".pkl") for est in estimators}
for f in est_fe_files.values():
    assert os.path.exists(f), f + " does not exist."

est_fes = {est : pickle.load( open(est_fe_files[est]) ) for est in estimators}

for est in estimators:
    est_fes[est]["fes"] = est_fes[est]["fes"].values()
    est_fes[est]["mean"] = _match_min(est_fes[est]["mean"], num_fe)


# plot fe
xs = [equal_stride_subsample(pulling_times, args.nsamples) for _ in estimators]
xs.append(equal_stride_subsample(pulling_times, args.nsamples))

ys = [equal_stride_subsample(est_fes[est]["mean"], args.nsamples) for est in estimators]
ys.append(equal_stride_subsample(num_fe, args.nsamples))

yerrs = [equal_stride_subsample(est_fes[est]["std"], args.nsamples) for est in estimators]
yerrs = [yerr/2. for yerr in yerrs] # 1 std
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

plot_lines(xs, ys, yerrs=yerrs,
           xlabel=args.xlabel, ylabel=args.ylabel_fe,
           out=args.fe_out,
           legends=estimators + ["num"],
           markers=MARKERS,
           legend_pos="best",
           legend_fontsize=8,
           xlimits=xlimits,
           ylimits=ylimits_fe,
           lw=1.0,
           markersize=5,
           alpha=0.5,
           n_xtics=8,
           n_ytics=8)


# plot RMSE
xs = [equal_stride_subsample(pulling_times, args.nsamples) for _ in estimators]

ys = [_rmse(est_fes[est]["fes"], num_fe) for est in estimators]
ys = [equal_stride_subsample(y, args.nsamples) for y in ys]

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


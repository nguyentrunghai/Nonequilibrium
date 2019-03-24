"""
"""

from __future__ import print_function
from __future__ import division

#import seaborn as sns
#sns.set()

import argparse
import pickle
import os
import copy

import numpy as np

from utils import bin_centers
from _plots import plot_lines

from _fe_pmf_plot_utils import first_to_zero, min_to_zero, reverse_data_order, replicate_data_cuc7_da
from _fe_pmf_plot_utils import put_first_of_fe_to_zero, put_argmin_of_pmf_to_target
from _fe_pmf_plot_utils import replicate

parser = argparse.ArgumentParser()

parser.add_argument("--pull_data_dir", type=str, default="./")

parser.add_argument("--free_energies_pmfs_files", type=str, default="file1 file2")
parser.add_argument("--us_fe_file", type=str, default="fe_us.pkl")
parser.add_argument("--us_pmf_file", type=str, default="pmf_us.dat")

parser.add_argument("--system_type", type=str, default="symmetric")

parser.add_argument("--data_estimator_pairs", type=str, default="s_u s_b s_s f_u r_u fr_b")

parser.add_argument("--n_fe_points_to_plot", type=int, default=0)

parser.add_argument("--fe_xlabel", type=str, default="$\lambda$ (nm)")
parser.add_argument("--fe_ylabel", type=str, default="RMSE[$\Delta F_{\lambda}$] (RT)")

parser.add_argument("--pmf_xlabel", type=str, default="$d$ (nm)")
parser.add_argument("--pmf_ylabel", type=str, default="RMSE[$\Phi(d)$] (RT)")
# for symmetric data plot the pmf from pmf[bin_ind_to_start_to_plot] to pmf[len - bin_ind_to_start_to_plot]
# for asymmetric data plot pmf from pmf[bin_ind_to_start_to_plot] to pmf[len]
parser.add_argument("--bin_ind_to_start_to_plot", type=int, default=0)

parser.add_argument("--legend_ncol_fe", type=int, default=1)
parser.add_argument("--legend_ncol_pmf", type=int, default=1)

parser.add_argument("--xlimits_fe", type=str, default="None")
parser.add_argument("--ylimits_fe", type=str, default="None")

parser.add_argument("--xlimits_pmf", type=str, default="None")
parser.add_argument("--ylimits_pmf", type=str, default="None")

parser.add_argument("--fe_out", type=str, default="fe_rmse.pdf")
parser.add_argument("--pmf_out", type=str, default="pmf_rmse.pdf")

args = parser.parse_args()

KB = 0.0019872041   # kcal/mol/K
TEMPERATURE = 300.
BETA = 1/KB/TEMPERATURE


def _rmse(main_estimates, reference):
    squared_deviations = [(estimates - reference)**2 for estimates in main_estimates.values()]
    squared_deviations = np.array(squared_deviations)
    return np.sqrt(squared_deviations.mean(axis=0))


def _rmse_std_error(pull_data, reference):
    bootstrap_keys = [bt for bt in pull_data if bt.startswith("bootstrap_")]
    bootstrap_estimates = [_rmse(pull_data[bootstrap_key], reference) for bootstrap_key in bootstrap_keys]
    bootstrap_estimates = np.array(bootstrap_estimates)
    return bootstrap_estimates.std(axis=0)


def _down_sampling(array, n_points):
    indices = np.linspace(0, array.shape[0] - 1, n_points)
    indices = np.round(indices)
    indices = indices.astype(int)
    return indices


free_energies_pmfs_files = [os.path.join(args.pull_data_dir, f) for f in args.free_energies_pmfs_files.split()]
print("free_energies_pmfs_files", free_energies_pmfs_files)

print("us_fe_file", args.us_fe_file)
print("us_pmf_file", args.us_pmf_file)

fe_us = pickle.load(open(args.us_fe_file, "r"))
fe_us["fe"] = first_to_zero(fe_us["fe"])
if args.system_type == "asymmetric":
    fe_us["lambdas"] = replicate(fe_us["lambdas"], "as_is", exclude_last_in_first_half=True)
    fe_us["fe"] = replicate(fe_us["fe"], "as_is", exclude_last_in_first_half=True)

pmf_us = dict()
pmf_us["pmf"] = np.loadtxt(args.us_pmf_file)[:, 1]
pmf_us["pmf"] *= BETA   # kcal/mol to KT/mol or RT
pmf_us["pmf"] = min_to_zero(pmf_us["pmf"])

data_estimator_pairs = args.data_estimator_pairs.split()
if len(data_estimator_pairs) != len(free_energies_pmfs_files):
    raise ValueError("data_estimator_pairs and free_energies_pmfs_files not consistent")

for label in data_estimator_pairs:
    if label not in ["s_u", "s_b", "s_s", "f_u", "r_u", "fr_b"]:
        raise ValueError("Unrecognized label: " + label)

free_energies = {}
pmfs = {}

all_data = {}
for file_name, label in zip(free_energies_pmfs_files, data_estimator_pairs):
    data = pickle.load(open(file_name, "r"))

    if label == "s_u":
        pmf_us["pmf_bin_edges"] = copy.deepcopy(data["pmfs"]["pmf_bin_edges"])

    # reverse order
    if label == "r_u":
        data = reverse_data_order(data)

    # replicate data
    if label in ["f_u", "r_u", "fr_b"]:
        print("Right replicate for", label)
        data = replicate_data_cuc7_da(data, args.system_type)

    # put first of fes to zero
    data = put_first_of_fe_to_zero(data)

    # put argmin of pmf to pmf_exact["pmf"]
    data = put_argmin_of_pmf_to_target(data, pmf_us["pmf"])

    all_data[label] = data


fe_rmse = {}
fe_rmse_std_error = {}
pmf_rmse = {}
pmf_rmse_std_error = {}
for label in data_estimator_pairs:
    fe_rmse[label] = _rmse(all_data[label]["free_energies"]["main_estimates"], fe_us["fe"])
    fe_rmse_std_error[label] = _rmse_std_error(all_data[label]["free_energies"], fe_us["fe"])

    pmf_rmse[label] = _rmse(all_data[label]["pmfs"]["main_estimates"], pmf_us["pmf"])
    pmf_rmse_std_error[label] = _rmse_std_error(all_data[label]["pmfs"], pmf_us["pmf"])

# TODO
# plot fe rmse
xs = []
ys = []
yerrs = []
for label in data_estimator_pairs:
    x = all_data[label]["free_energies"]["lambdas"]
    y = fe_rmse[label]
    yerr = fe_rmse_std_error[label] / 2

    if args.n_fe_points_to_plot != 0:
        indices = _down_sampling(x, args.n_fe_points_to_plot)
        x = x[indices]
        y = y[indices]
        yerr = yerr[indices]

    xs.append(x)
    ys.append(y)
    yerrs.append(yerr)

if args.xlimits_fe.lower() != "none":
    xlimits_fe = [float(s) for s in args.xlimits_fe.split()]
else:
    xlimits_fe = None

if args.ylimits_fe.lower() != "none":
    ylimits_fe = [float(s) for s in args.ylimits_fe.split()]
else:
    ylimits_fe = None

MARKERS = ["<", ">", "^", "v", "s", "d", "."]

plot_lines(xs, ys, yerrs=yerrs,
           xlabel=args.fe_xlabel, ylabel=args.fe_ylabel,
           out=args.fe_out,
           legends=data_estimator_pairs,
           legend_pos="best",
           legend_ncol=args.legend_ncol_fe,
           legend_fontsize=8,
           markers=MARKERS,
           xlimits=xlimits_fe,
           ylimits=ylimits_fe,
           lw=1.0,
           markersize=4,
           alpha=1.,
           n_xtics=8,
           n_ytics=8)


# plot pmf rmse
start_pmf_ind = args.bin_ind_to_start_to_plot

xs = []
ys = []
yerrs = []
for label in data_estimator_pairs:
    x = bin_centers(all_data[label]["pmfs"]["pmf_bin_edges"])

    end_pmf_ind = len(x) - start_pmf_ind
    xs.append(x[start_pmf_ind : end_pmf_ind])
    ys.append(pmf_rmse[label][start_pmf_ind : end_pmf_ind])
    yerrs.append(pmf_rmse_std_error[label][start_pmf_ind : end_pmf_ind] / 2)

if args.xlimits_pmf.lower() != "none":
    xlimits_pmf = [float(s) for s in args.xlimits_pmf.split()]
else:
    xlimits_pmf = None

if args.ylimits_pmf.lower() != "none":
    ylimits_pmf = [float(s) for s in args.ylimits_pmf.split()]
else:
    ylimits_pmf = None

plot_lines(xs, ys, yerrs=yerrs,
           xlabel=args.pmf_xlabel, ylabel=args.pmf_ylabel,
           out=args.pmf_out,
           legends=data_estimator_pairs,
           legend_ncol=args.legend_ncol_pmf,
           legend_pos="best",
           legend_fontsize=8,
           markers=MARKERS,
           xlimits=xlimits_pmf,
           ylimits=ylimits_pmf,
           lw=1.0,
           markersize=4,
           alpha=1.,
           n_xtics=8,
           n_ytics=8)
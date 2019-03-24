"""
"""

from __future__ import print_function
from __future__ import division

import os
import argparse
import glob
import pickle

import numpy as np

from _plots import plot_lines
from _fe_pmf_plot_utils import first_to_zero, min_to_zero
from _fe_pmf_plot_utils import reverse_data_order, replicate_data_1d
from _fe_pmf_plot_utils import put_first_of_fe_to_zero, put_argmin_of_pmf_to_target

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./")

parser.add_argument("--fes_pmfs_file_matching", type=str, default="file1* file2*")
parser.add_argument("--num_fe_file", type=str, default="fe_numerical.pkl")
parser.add_argument("--exact_pmf_file", type=str, default="pmf_exact.pkl")

parser.add_argument( "--system_type", type=str, default="symmetric")

parser.add_argument("--data_estimator_pairs", type=str, default="s_u s_b s_s f_u r_u fr_b")

parser.add_argument( "--pmf_bin_truncate", type=int, default=0)

parser.add_argument("--fe_xlabel", type=str, default="# of trajectories")
parser.add_argument("--fe_ylabel", type=str, default="RMSE[$\Delta F_{\lambda}$]")

parser.add_argument("--pmf_xlabel", type=str, default="# of trajectories")
parser.add_argument("--pmf_ylabel", type=str, default="RMSE[$\Phi(z)$]")

parser.add_argument("--legend_ncol_fe", type=int, default=1)
parser.add_argument("--legend_ncol_pmf", type=int, default=1)

parser.add_argument("--xlimits_fe", type=str, default="None")
parser.add_argument("--ylimits_fe", type=str, default="None")

parser.add_argument("--xlimits_pmf", type=str, default="None")
parser.add_argument("--ylimits_pmf", type=str, default="None")

parser.add_argument("--fe_out", type=str, default="fe_rmse_time_depend.pdf")
parser.add_argument("--pmf_out", type=str, default="pmf_rmse_time_depend.pdf")

args = parser.parse_args()


def _rmse(main_estimates, reference, truncate):
    begin = truncate
    end = len(reference) - truncate
    #squared_deviations = [(estimates[begin : end] - reference[begin : end])**2
    #                      for estimates in main_estimates.values()]

    squared_deviations = []
    for estimates in main_estimates.values():
        sd = (estimates[begin : end] - reference[begin : end])**2
        sd = sd[np.abs(sd) != np.inf]
        squared_deviations.append(np.nanmean(sd))

    squared_deviations = np.array(squared_deviations)
    return np.sqrt(squared_deviations.mean())


def _rmse_std_error(pull_data, reference, truncate):
    bootstrap_keys = [bt for bt in pull_data if bt.startswith("bootstrap_")]
    bootstrap_estimates = [_rmse(pull_data[bootstrap_key], reference, truncate) for bootstrap_key in bootstrap_keys]
    bootstrap_estimates = np.array(bootstrap_estimates)
    return bootstrap_estimates.std(axis=0)


fes_pmfs_file_matching = args.fes_pmfs_file_matching.split()
print("fes_pmfs_file_matching", fes_pmfs_file_matching)

data_estimator_pairs = args.data_estimator_pairs.split()
print("data_estimator_pairs", data_estimator_pairs)

if len(fes_pmfs_file_matching) != len(data_estimator_pairs):
    raise ValueError("len of fes_pmfs_file_matching not the same as data_estimator_pair")

for label in data_estimator_pairs:
    if label not in ["s_u", "s_b", "s_s", "f_u", "r_u", "fr_b"]:
        raise ValueError("Unrecognized label: " + label)

fes_pmfs_files = {}
for label, matching in zip(data_estimator_pairs, fes_pmfs_file_matching):
    fes_pmfs_files[label] = glob.glob(os.path.join(args.data_dir, matching))

number_of_files = [len(files) for files in fes_pmfs_files.values()]
if np.unique(number_of_files).shape[0] != 1:
    raise ValueError("different labels do not have the same number of files")

num_fe_file = os.path.join(args.data_dir, args.num_fe_file)
print("num_fe_file", num_fe_file)

exact_pmf_file = os.path.join(args.data_dir, args.exact_pmf_file)
print("exact_pmf_file", exact_pmf_file)

fe_num = pickle.load(open(num_fe_file, "r"))
fe_num["fe"] = first_to_zero(fe_num["fe"])

pmf_exact = pickle.load(open(exact_pmf_file , "r"))
pmf_exact["pmf"] = min_to_zero(pmf_exact["pmf"])

fe_rmse  = {}
pmf_rmse = {}
for label in data_estimator_pairs:
    print("label", label)
    fe_rmse[label] = []
    pmf_rmse[label] = []

    for fes_pmfs_file in fes_pmfs_files[label]:
        print("processing file", fes_pmfs_file)
        data = pickle.load(open(fes_pmfs_file, "r"))

        if data["free_energies"]["ntrajs_per_block"] != data["pmfs"]["ntrajs_per_block"]:
            raise ValueError("ntrajs_per_block are not consistent in" + fes_pmfs_file)
        ntrajs = data["free_energies"]["ntrajs_per_block"]
        if label in ["s_u", "s_b", "s_s"]:
            ntrajs *= 2

        # reverse order
        if label == "r_u":
            data = reverse_data_order(data)

        # replicate data
        if label in ["f_u", "r_u", "fr_b"]:
            data = replicate_data_1d(data, args.system_type)

        # put first of fes to zero
        data = put_first_of_fe_to_zero(data)

        # put argmin of pmf to pmf_exact["pmf"]
        data = put_argmin_of_pmf_to_target(data, pmf_exact["pmf"])

        fe_r = _rmse(data["free_energies"]["main_estimates"], fe_num["fe"], truncate=0)
        fe_r_error = _rmse_std_error(data["free_energies"], fe_num["fe"], truncate=0)
        fe_rmse[label].append((ntrajs, fe_r, fe_r_error))

        pmf_r = _rmse(data["pmfs"]["main_estimates"], pmf_exact["pmf"], truncate=args.pmf_bin_truncate)
        pmf_r_error = _rmse_std_error(data["pmfs"], pmf_exact["pmf"], truncate=args.pmf_bin_truncate)
        pmf_rmse[label].append((ntrajs, pmf_r, pmf_r_error))

    fe_rmse[label].sort(key=lambda item: item[0])
    pmf_rmse[label].sort(key=lambda item: item[0])

# plot fe rmse
xs = []
ys = []
yerrs = []
for label in data_estimator_pairs:
    x = np.array([item[0] for item in fe_rmse[label]])
    y = np.array([item[1] for item in fe_rmse[label]])
    yerr = np.array([item[2] for item in fe_rmse[label]])

    xs.append(x)
    ys.append(y)
    yerrs.append(yerr / 2)

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
xs = []
ys = []
yerrs = []
for label in data_estimator_pairs:
    x = np.array([item[0] for item in pmf_rmse[label]])
    y = np.array([item[1] for item in pmf_rmse[label]])
    yerr = np.array([item[2] for item in pmf_rmse[label]])

    xs.append(x)
    ys.append(y)
    yerrs.append(yerr / 2)

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
